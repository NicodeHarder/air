from collections import OrderedDict
import gc

import torch

from lxt.utils import one_hot_max, one_hot, ModelLayerUtils

from lxt.modules import LinearEpsilon
from lxt.rules import EpsilonRule
from models.model import *


class LatentRelevanceAttributor:

    def __init__(self, layers_list_to_track) -> None:
        self.layers_list_to_track = layers_list_to_track
        self.latent_output = {}

    def lrp_pass(self, model, inputs, targets, composite, initial_relevance, device, args):

        if initial_relevance == 1:
            initial_relevance_function = one_hot
        elif initial_relevance == "logit":
            initial_relevance_function = one_hot_max

        with torch.enable_grad():

            if inputs.requires_grad == False:
                inputs.requires_grad = True

            self.compute_relevance(model, inputs, targets, initial_relevance_function, device, args)

        self.remove_hooks()

        return

    def predict(self, model, inputs, device="cpu"):

        # Ensure inputs require gradients for backward pass
        if not inputs.requires_grad:
            inputs = inputs.requires_grad_()

        output_logits = model(inputs_embeds=inputs, use_cache=False).logits
        max_logits, max_indices = torch.max(output_logits[0, -1, :], dim=-1)

        return max_logits

    def compute_relevance(
            self, model, inputs, targets, initial_relevance_function, device, args
    ):

        self.clear_latent_info()
        self.hook_handles = self.register_hooks(model)

        input_device = model.get_input_embeddings().weight.device
        inputs = inputs.to(input_device)
        output = self.predict(model, inputs, input_device)

        (relevance,) = torch.autograd.grad(
            outputs=output,
            inputs=inputs,
            grad_outputs=torch.ones_like(output) * output.detach(),
            retain_graph=False,
        )

        print(f'\nInputs [Sum, Shape, Device] = {inputs.sum()}, {inputs.shape}, {inputs.device}')
        print(f'Outputs [Sum, Shape, Device] = {output.sum()}, {output.shape}, {output.device}')
        print(f'Relevance [Sum, Shape, Device] = {relevance.sum()}, {relevance.shape}, {relevance.device}')

        del output
        torch.cuda.empty_cache()

        return

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()


class LXTLatentRelevanceAttributor(LatentRelevanceAttributor):

    def __init__(self, layers_list_to_track) -> None:

        super().__init__(layers_list_to_track)

        self.get_latent_hidden_states = {}
        self.set_latent_hidden_states = {}

        self.get_latent_activations = {}
        self.set_latent_activations = {}

        self.get_hidden_states_relevances = {}
        self.get_weight_relevances = {}
        self.get_activation_relevances = {}

    def register_hooks(self, model):
        hook_handles = []
        for layer_name in self.layers_list_to_track:
            for name, module in model.named_modules():
                if name == layer_name:
                    hook_handles.append(module.register_backward_hook(self.backward_get_hook_wrapper(layer_name)))

        return hook_handles

    def backward_get_hook_wrapper(self, layer_name):
        def get_relevance(module, in_gradient, out_gradient):
            
            self.get_hidden_states_relevances[layer_name] = in_gradient[0].detach().cpu()
            self.get_weight_relevances[layer_name] = module.module.weight.relevance.detach().cpu()
            self.get_activation_relevances[layer_name] = out_gradient[0].detach().cpu()

            if hasattr(module.module.weight, 'relevance'):
                del module.module.weight.relevance

            torch.cuda.empty_cache()

        return get_relevance

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def clear_latent_info(self):
        self.get_latent_hidden_states = {}
        self.get_latent_activations = {}
        self.get_hidden_states_relevances = {}
        self.get_activation_relevances = {}


class ComponentAttribution:

    def __init__(self, attribution_type, model_type, target_layer_type):
        self.model_type = model_type
        self.attribution_type = attribution_type
        self.target_layer_type = target_layer_type
        self.attributor = self.choose_attributor()

    def choose_attributor(self):
        attributor = LXTLatentRelevanceAttributor
        return attributor

    @staticmethod
    def get_layer_names(model, target_layer_type):
        layer_names = ModelLayerUtils.get_layer_names(model, target_layer_type)

        if target_layer_type == torch.nn.Softmax:
            layer_names = [name.replace(".module", "") for name in layer_names]

        if target_layer_type in [torch.nn.Linear or LinearEpsilon or EpsilonRule]:
            layer_names = layer_names[:-1]

        return layer_names

    def attribute(
            self,
            args,
            model,
            dataloader,
            attribution_composite,
            abs_flag=True,
            filter_layers_by="mlp",
            device="cpu",
            save_paths=''
    ):

        print(f"\n\nLXT: Computing Relevances ... ")

        model.eval()

        self.layer_names = ComponentAttribution.get_layer_names(model, self.target_layer_type)

        if filter_layers_by == "mlp":
            self.layer_names = [name for name in self.layer_names if "mlp" in name]

        for layer, _ in model.state_dict().items():
            if ('proj.module.weight.u' in layer or 'proj.module.weight.sigma' in layer or 'proj.module.weight.v' in layer) and layer not in self.layer_names:
                self.layer_names.append(layer)

        attributor = self.attributor(self.layer_names)

        sum_hidden_states_relevances = OrderedDict([])
        sum_weight_relevances = OrderedDict([])
        sum_activation_relevances = OrderedDict([])
        counter = 0
        log_interval = 1 # 10

        for sample_idx, inputs in enumerate(dataloader):

            counter += 1
            print(f"\nLXT: Processing Sample {counter}/{len(dataloader)} (incremental accumulation)...")
                # Print memory usage
                # if torch.cuda.is_available():
                #     print(f"    GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

            cut_idx = min(args.relevance_seq_len, int(inputs['attention_mask'].sum()))  # consider for reasoning data at smaller sequence lengths indicated by attention masks

            inputs = inputs['input_ids']

            if len(inputs.shape) == 1:
                inputs = inputs.unsqueeze(0)

            inputs = inputs[:, :cut_idx]

            inputs = inputs.to(model.device)
            input_embed = get_input_embeddings(model, inputs)  # .float()

            for name, param in model.named_parameters():
                if hasattr(param, 'relevance'):
                    delattr(param, 'relevance')

            model.zero_grad(set_to_none=True)

            torch.cuda.empty_cache()
            gc.collect()

            attributor.lrp_pass(
                model,
                input_embed,
                None,
                composite=None,
                initial_relevance=1,
                device=device,
                args=args
            )

            del inputs, input_embed
            # Only clear CUDA cache once per sample if needed
            torch.cuda.empty_cache()

            # --- In-memory accumulation ---
            for layer_name in self.layer_names:
                if 'head' in layer_name:
                    continue

                # Accumulate Weight Relevances
                weight_rel = torch.abs(attributor.get_weight_relevances[layer_name].detach().cpu())
                weight_rel /= len(dataloader)
                
                if layer_name not in sum_weight_relevances:
                    sum_weight_relevances[layer_name] = weight_rel
                else:
                    sum_weight_relevances[layer_name] += weight_rel

                # Accumulate Hidden States Relevances (if needed)
                if args.IMPORTANCE_AWARE_PROFILING_MATRIX:
                    hidden_rel = torch.abs(attributor.get_hidden_states_relevances[layer_name].detach().cpu())[0]
                    hidden_rel /= len(dataloader)
                    if layer_name not in sum_hidden_states_relevances:
                        sum_hidden_states_relevances[layer_name] = hidden_rel
                    else:
                        sum_hidden_states_relevances[layer_name] += hidden_rel

                # Accumulate Activation Relevances (if needed)
                if args.COMPUTE_RELEVANCES_ACTIVATIONS:
                    act_rel = torch.abs(attributor.get_activation_relevances[layer_name].detach().cpu())[0]
                    act_rel /= len(dataloader)
                    if layer_name not in sum_activation_relevances:
                        sum_activation_relevances[layer_name] = act_rel
                    else:
                        sum_activation_relevances[layer_name] += act_rel
            attributor.clear_latent_info()

            if args.EARLY_EXIT:
                break

        print(f"\nLXT: Saving accumulated relevances to disk...")
        layer_block_index = 0
        current_block_data_weights = {}
        current_block_data_hidden = {}
        current_block_data_act = {}
        
        for layer_name in self.layer_names:
            if 'head' in layer_name:
                continue

            if f'attn.q' in layer_name and current_block_data_weights:
                
                torch.save(current_block_data_weights, str(args.save_paths_relevances[1] + f'layer_{layer_block_index}.pt'))
                if args.IMPORTANCE_AWARE_PROFILING_MATRIX:
                    torch.save(current_block_data_hidden, str(args.save_paths_relevances[0] + f'layer_{layer_block_index}.pt'))
                if args.COMPUTE_RELEVANCES_ACTIVATIONS:
                    torch.save(current_block_data_act, str(args.save_paths_relevances[2] + f'layer_{layer_block_index}.pt'))
                
                current_block_data_weights = {}
                current_block_data_hidden = {}
                current_block_data_act = {}
                layer_block_index += 1
                print(f"  -> Saved layer block {layer_block_index}/{model.config.num_hidden_layers}")

            current_block_data_weights[layer_name] = sum_weight_relevances[layer_name]
            if args.IMPORTANCE_AWARE_PROFILING_MATRIX:
                current_block_data_hidden[layer_name] = sum_hidden_states_relevances[layer_name]
            if args.COMPUTE_RELEVANCES_ACTIVATIONS:
                current_block_data_act[layer_name] = sum_activation_relevances[layer_name]
            
        if current_block_data_weights:
            torch.save(current_block_data_weights, str(args.save_paths_relevances[1] + f'layer_{layer_block_index}.pt'))
            if args.IMPORTANCE_AWARE_PROFILING_MATRIX:
                torch.save(current_block_data_hidden, str(args.save_paths_relevances[0] + f'layer_{layer_block_index}.pt'))
            if args.COMPUTE_RELEVANCES_ACTIVATIONS:
                torch.save(current_block_data_act, str(args.save_paths_relevances[2] + f'layer_{layer_block_index}.pt'))
            print(f"  -> Saved final layer block {layer_block_index + 1}")

        return


def get_input_embeddings(model, input_ids):
    input_embeds = model.get_input_embeddings()(input_ids)

    if input_embeds.requires_grad == False:
        input_embeds.requires_grads = True

    return input_embeds