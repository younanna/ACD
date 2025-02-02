import warnings
import inspect

import torch
from torch.nn import functional as F
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional, Callable, List, Union, TYPE_CHECKING
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.generation.utils import (
    GenerateOutput, 
    GenerateNonBeamOutput, 
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    NEED_SETUP_CACHE_CLASSES_MAPPING, 
    QUANT_BACKEND_CLASSES_MAPPING)
from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.generation.logits_process import LogitsProcessorList    
from transformers.generation.stopping_criteria import StoppingCriteriaList

from transformers.utils import (
    ModelOutput,
    is_hqq_available,
    is_quanto_available,
    is_torchdynamo_compiling,
    logging,
)
from transformers.cache_utils import (
    Cache,
    DynamicCache,
    EncoderDecoderCache,
    QuantizedCacheConfig,
    OffloadedCache,
)

from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import BeamSearchScorer, ConstrainedBeamSearchScorer

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.streamers import BaseStreamer
    
logger = logging.get_logger(__name__)




class LMWrapper:
    def __init__(
        self,
        model_name_or_path: str,
        cache_dir: str,
        device_map: str,
        max_memory: dict,
        offload_folder: str = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.gen_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            cache_dir=cache_dir,
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            # quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map=device_map, max_memory=max_memory, offload_folder=offload_folder
        ).eval()
        self.model_name_or_path = model_name_or_path
        self.device = self.gen_model.device
        

    @torch.no_grad()
    def generate_contrast(
        self,
        inputs: Optional[torch.Tensor] = None,
        inputs_pos: Optional[torch.Tensor] = None, # added
        inputs_neg: Optional[torch.Tensor] = None, # added
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
        main_task: Optional[str] = 'generation',
        dummy: Optional[list] = None,
        **kwargs,
        ) -> Union[GenerateOutput, torch.LongTensor]:
        r"""

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config ([`~generation.GenerationConfig`], *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complements the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
                sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
                intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
                `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
                generating before other GPUs. Otherwise it'll be set to `False`.
            assistant_model (`PreTrainedModel`, *optional*):
                An assistant model that can be used to accelerate generation. The assistant model must have the exact
                same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
                is much faster than running generation with the model you're calling generate from. As such, the
                assistant model should be much smaller.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The negative prompt needed for some processors such as CFG. The batch size must match the input batch
                size. This is an experimental feature, subject to breaking API changes in future versions.
            negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention_mask for `negative_prompt_ids`.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],
                    - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call

        self.gen_model._validate_model_class()
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        generation_config, model_kwargs = self.gen_model._prepare_generation_config(generation_config, **kwargs)
        if inputs_pos is not None: model_kwargs_pos = model_kwargs.copy()
        if inputs_neg is not None: model_kwargs_neg = model_kwargs.copy()
        self.gen_model._validate_model_kwargs(model_kwargs.copy())
        self.gen_model._validate_assistant(assistant_model)

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.gen_model.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        # inputs_tensor: [batch_size, seq_len]
        inputs_tensor, model_input_name, model_kwargs = self.gen_model._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        if inputs_pos is not None:
            inputs_tensor_pos, _, model_kwargs_pos = self.gen_model._prepare_model_inputs(
                inputs_pos, generation_config.bos_token_id, model_kwargs_pos
            )
        if inputs_neg is not None:
            inputs_tensor_neg, _, model_kwargs_neg = self.gen_model._prepare_model_inputs(
                inputs_neg, generation_config.bos_token_id, model_kwargs_neg
            )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self.gen_model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.gen_model.config.is_encoder_decoder and not is_torchdynamo_compiling():
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.gen_model.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
            if inputs_pos is not None: model_kwargs_pos["use_cache"] = True
            if inputs_neg is not None: model_kwargs_neg["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache
            if inputs_pos is not None: model_kwargs_pos["use_cache"] = generation_config.use_cache
            if inputs_neg is not None: model_kwargs_neg["use_cache"] = generation_config.use_cache

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self.gen_model._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
            )
            if inputs_pos is not None:
                model_kwargs_pos["attention_mask"] = self.gen_model._prepare_attention_mask_for_generation(
                    inputs_tensor_pos, generation_config._pad_token_tensor, generation_config._eos_token_tensor
                )
            if inputs_neg is not None:
                model_kwargs_neg["attention_mask"] = self.gen_model._prepare_attention_mask_for_generation(
                    inputs_tensor_neg, generation_config._pad_token_tensor, generation_config._eos_token_tensor
                )

        if self.gen_model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self.gen_model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )
            if inputs_pos is not None:
                model_kwargs_pos = self.gen_model._prepare_encoder_decoder_kwargs_for_generation(
                    inputs_tensor_pos, model_kwargs_pos, model_input_name, generation_config
                )
            if inputs_neg is not None:
                model_kwargs_neg = self.gen_model._prepare_encoder_decoder_kwargs_for_generation(
                    inputs_tensor_neg, model_kwargs_neg, model_input_name, generation_config
                )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.gen_model.config.is_encoder_decoder:
            input_ids, model_kwargs = self.gen_model._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
            if inputs_pos is not None:
                input_ids_pos, model_kwargs_pos = self.gen_model._prepare_decoder_input_ids_for_generation(
                    batch_size=batch_size,
                    model_input_name=model_input_name,
                    model_kwargs=model_kwargs_pos,
                    decoder_start_token_id=generation_config._decoder_start_token_tensor,
                    device=inputs_tensor_pos.device,
                )
            if inputs_neg is not None:
                input_ids_neg, model_kwargs_neg = self.gen_model._prepare_decoder_input_ids_for_generation(
                    batch_size=batch_size,
                    model_input_name=model_input_name,
                    model_kwargs=model_kwargs_neg,
                    decoder_start_token_id=generation_config._decoder_start_token_tensor,
                    device=inputs_tensor_neg.device,
                )

        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")
            if inputs_pos is not None: input_ids_pos = inputs_tensor_pos if model_input_name == "input_ids" else model_kwargs_pos.pop("input_ids")
            if inputs_neg is not None: input_ids_neg = inputs_tensor_neg if model_input_name == "input_ids" else model_kwargs_neg.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.gen_model.heal_tokens(input_ids, tokenizer)
            if inputs_pos is not None: input_ids_pos = self.gen_model.heal_tokens(input_ids_pos, tokenizer)
            if inputs_neg is not None: input_ids_neg = self.gen_model.heal_tokens(input_ids_neg, tokenizer)


        if streamer is not None:
            streamer.put(input_ids.cpu())
            if inputs_pos is not None: streamer.put(input_ids_pos.cpu())
            if inputs_neg is not None: streamer.put(input_ids_neg.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        if inputs_pos is not None: input_ids_length_pos = input_ids_pos.shape[-1]
        if inputs_neg is not None: input_ids_length_neg = input_ids_neg.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self.gen_model._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        use_dynamic_cache_by_default = False
        if "mamba" in self.gen_model.__class__.__name__.lower():
            cache_name = "cache_params"
        else:
            cache_name = "past_key_values"

        # TODO(joao): support static caches in assisted generation. assisted generation needs to roll back caches,
        # which is only supported in dynamic caches atm
        if (
            assistant_model is not None
            and generation_config.cache_implementation is not None
            and self.gen_model._supports_default_dynamic_cache()
        ):
            logger.warning_once(
                "An assistant model is provided, using a dynamic cache instead of a cache of type="
                f"'{generation_config.cache_implementation}'."
            )
            generation_config.cache_implementation = None

        if (model_kwargs.get(cache_name) is not None) and is_torchdynamo_compiling():
            raise ValueError(
                "Passing `past_key_values` is not supported when compiling `model.generate` with torch.compile -- you "
                "may get incorrect outputs. Please compile `model.forward` only or use the `cache_implementation` "
                "input argument."
            )
        if generation_config.cache_implementation is not None and (model_kwargs.get(cache_name) is not None):
            raise ValueError(
                f"Passing both `cache_implementation` (used to initialize certain caches) and `{cache_name}` (a "
                "Cache object) is unsupported. Please use only one of the two."
            )
        elif generation_config.cache_implementation is not None:
            if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
                if generation_config.cache_implementation == "static" and not self.gen_model._supports_static_cache:
                    raise ValueError(
                        "This model does not support `cache_implementation='static'`. Please check the following "
                        "issue: https://github.com/huggingface/transformers/issues/28981"
                    )
                model_kwargs[cache_name] = self.gen_model._get_cache(
                    cache_implementation=generation_config.cache_implementation,
                    max_batch_size=generation_config.num_beams * generation_config.num_return_sequences * batch_size,
                    max_cache_len=generation_config.max_length,
                    device=device,
                    model_kwargs=model_kwargs,
                )
            elif generation_config.cache_implementation == "quantized":
                if not self.gen_model._supports_quantized_cache:
                    raise ValueError(
                        "This model does not support the quantized cache. If you want your model to support quantized "
                        "cache, please open an issue."
                    )

                cache_config = (
                    generation_config.cache_config
                    if generation_config.cache_config is not None
                    else QuantizedCacheConfig()
                )
                cache_class = QUANT_BACKEND_CLASSES_MAPPING[cache_config.backend]

                if cache_config.backend == "quanto" and not is_quanto_available():
                    raise ImportError(
                        "You need to install `quanto` in order to use KV cache quantization with quanto backend. "
                        "Please install it via  with `pip install quanto`"
                    )
                elif cache_config.backend == "HQQ" and not is_hqq_available():
                    raise ImportError(
                        "You need to install `HQQ` in order to use KV cache quantization with HQQ backend. "
                        "Please install it via  with `pip install hqq`"
                    )

                model_kwargs[cache_name] = cache_class(cache_config)
            elif generation_config.cache_implementation == "offloaded":
                model_kwargs[cache_name] = OffloadedCache()
        # Use DynamicCache() instance by default. This will avoid back and forth from legacy format that
        # keeps copying the cache thus using much more memory
        elif generation_config.cache_implementation is None and self.gen_model._supports_default_dynamic_cache():
            past = model_kwargs.get(cache_name, None)
            requires_cross_attention_cache = (
                self.gen_model.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
            )
            if past is None:
                model_kwargs[cache_name] = (
                    DynamicCache()
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache(DynamicCache(), DynamicCache())
                )
                use_dynamic_cache_by_default = True
            elif isinstance(past, tuple):
                model_kwargs[cache_name] = (
                    DynamicCache.from_legacy_cache(past)
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache.from_legacy_cache(past)
                )
                use_dynamic_cache_by_default = True

        self.gen_model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if not is_torchdynamo_compiling() and self.gen_model.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.gen_model.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.gen_model.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        prepared_logits_processor = self.gen_model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        if inputs_pos is not None:
            prepared_logits_processor_pos = self.gen_model._get_logits_processor(
                generation_config=generation_config,
                input_ids_seq_length=input_ids_length_pos,
                encoder_input_ids=inputs_tensor_pos,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                logits_processor=logits_processor,
                model_kwargs=model_kwargs_pos,
                negative_prompt_ids=negative_prompt_ids,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
            )
        if inputs_neg is not None:
            prepared_logits_processor_neg = self.gen_model._get_logits_processor(
                generation_config=generation_config,
                input_ids_seq_length=input_ids_length_neg,
                encoder_input_ids=inputs_tensor_neg,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                logits_processor=logits_processor,
                model_kwargs=model_kwargs_neg,
                negative_prompt_ids=negative_prompt_ids,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
            )

        # 9. prepare stopping criteria
        prepared_stopping_criteria = self.gen_model._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )

        # 10. go into different generation modes
        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if batch_size > 1:
                raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")
            if generation_config.cache_implementation == "static":
                raise ValueError("assisted generate is not supported with `static_cache`")
            if self.gen_model._is_stateful:
                # In assisted generation we need the ability to confirm whether the model would pick certain tokens,
                # which is not possible with stateful models (they can't reset to a previous subset of generated text)
                raise ValueError(
                    f"assisted generation is not supported with stateful models, such as {self.gen_model.__class__.__name__}"
                )

            # 11. Get the candidate generator, given the parameterization
            candidate_generator = self.gen_model._get_candidate_generator(
                generation_config=generation_config,
                input_ids=input_ids,
                inputs_tensor=inputs_tensor,
                assistant_model=assistant_model,
                logits_processor=logits_processor,
                model_kwargs=model_kwargs,
            )

            # 12. prepare logits warper (if `do_sample` is `True`)
            prepared_logits_warper = (
                self.gen_model._get_logits_warper(
                    generation_config,
                    device=input_ids.device,
                )
                if generation_config.do_sample
                else None
            )

            # 13. run assisted generate
            result = self.gen_model._assisted_decoding(
                input_ids,
                candidate_generator=candidate_generator,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        elif generation_mode == GenerationMode.DOLA_GENERATION:
            if self.gen_model._is_stateful:
                # DoLa decoding was not designed for stateful models, and would require some changes
                raise ValueError(
                    f"dola decoding is not supported with stateful models, such as {self.gen_model.__class__.__name__}"
                )
            prepared_logits_warper = (
                self.gen_model._get_logits_warper(generation_config, device=input_ids.device)
                if generation_config.do_sample
                else None
            )
            result = self.gen_model._dola_decoding(
                input_ids,
                dola_layers=generation_config.dola_layers,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")
            if self.gen_model._is_stateful:
                # Just like assisted generation, we need to be able to rollback to a previous state (see comment above)
                raise ValueError(
                    f"contrastive search is not supported with stateful models, such as {self.gen_model.__class__.__name__}"
                )

            result = self.gen_model._contrastive_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # 11. prepare logits warper
            prepared_logits_warper = (
                self.gen_model._get_logits_warper(generation_config, device=input_ids.device)
                if generation_config.do_sample
                else None
            )

            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self.gen_model._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.gen_model.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            result = self._my_sample(
                input_ids,
                input_ids_pos=input_ids_pos if inputs_pos is not None else None,
                input_ids_neg=input_ids_neg if inputs_neg is not None else None,
                logits_processor=prepared_logits_processor,
                logits_processor_pos=prepared_logits_processor_pos if inputs_pos is not None else None,
                logits_processor_neg=prepared_logits_processor_neg if inputs_neg is not None else None,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                model_kwargs_pos=model_kwargs_pos if inputs_pos is not None else None,
                model_kwargs_neg=model_kwargs_neg if inputs_neg is not None else None,
                mode=mode,
                main_task=main_task,
                dummy=dummy,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
            # 11. prepare logits warper
            prepared_logits_warper = (
                self.gen_model._get_logits_warper(generation_config, device=input_ids.device)
                if generation_config.do_sample
                else None
            )

            # 12. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )

            # 13. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self.gen_model._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.gen_model.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 14. run beam sample
            result = self.gen_model._beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self.gen_model._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.gen_model.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self.gen_model._group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                    not isinstance(generation_config.force_words_ids, list)
                    or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self.gen_model._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.gen_model.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self.gen_model._constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        # Convert to legacy cache if needed
        if use_dynamic_cache_by_default and generation_config.return_legacy_cache:
            if isinstance(result, ModelOutput) and hasattr(result, "past_key_values"):
                if isinstance(result.past_key_values, (DynamicCache, EncoderDecoderCache)):
                    result.past_key_values = result.past_key_values.to_legacy_cache()
        return result
            

    def _my_sample(
        self,
        input_ids: torch.LongTensor,
        input_ids_pos: Optional[torch.LongTensor] = None,
        input_ids_neg: Optional[torch.LongTensor] = None,
        logits_processor: LogitsProcessorList = None,
        logits_processor_pos: Optional[LogitsProcessorList] = None,
        logits_processor_neg: Optional[LogitsProcessorList] = None,
        stopping_criteria: StoppingCriteriaList = None,
        generation_config: GenerationConfig = None,
        synced_gpus: bool = None,
        streamer: Optional["BaseStreamer"] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        model_kwargs_pos: Optional[dict] = None,
        model_kwargs_neg: Optional[dict] = None,
        mode: Optional[str] = None, # 'naive',
        main_task: Optional[str] = 'generation',
        dummy: Optional[list] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                `generation_config`)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.gen_model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self.gen_model._get_initial_cache_position(input_ids, model_kwargs)
        if input_ids_pos is not None: model_kwargs_pos = self.gen_model._get_initial_cache_position(input_ids_pos, model_kwargs_pos)
        if input_ids_neg is not None: model_kwargs_neg = self.gen_model._get_initial_cache_position(input_ids_neg, model_kwargs_neg)

        # return anything for analysis -> if main_task.startswith('experiment'):
        # tensors: [n_batch, n_tokens, ...]
        ret_dict = {}
        
        # GREEDY_SEARCH_WHILE_LOOP
        while self.gen_model._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
        ):
            # prepare model inputs
            model_inputs = self.gen_model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            if input_ids_pos is not None: model_inputs_pos = self.gen_model.prepare_inputs_for_generation(input_ids_pos, **model_kwargs_pos)
            if input_ids_neg is not None: model_inputs_neg = self.gen_model.prepare_inputs_for_generation(input_ids_neg, **model_kwargs_neg)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
            if input_ids_pos is not None: 
                model_inputs_pos.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs_pos.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
            if input_ids_neg is not None: 
                model_inputs_neg.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs_neg.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            # forward pass to get next token
            outputs = self.gen_model(**model_inputs, return_dict=True)
            
            if input_ids_pos is not None:
                outputs_pos = self.gen_model(**model_inputs_pos, return_dict=True)
            if input_ids_neg is not None:
                outputs_neg = self.gen_model(**model_inputs_neg, return_dict=True)

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone()
            if input_ids_pos is not None: next_token_logits_pos = outputs_pos.logits[:, -1, :].clone()
            if input_ids_neg is not None: next_token_logits_neg = outputs_neg.logits[:, -1, :].clone()

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            if input_ids_pos is not None: next_token_scores_pos = logits_processor_pos(input_ids_pos, next_token_logits_pos)
            if input_ids_neg is not None: next_token_scores_neg = logits_processor_neg(input_ids_neg, next_token_logits_neg)
            # next_token_logits == next_token_scores
            
            if do_sample:
                next_token_scores = logits_warper(input_ids, next_token_scores)
                if input_ids_pos is not None: next_token_scores_pos = logits_warper(input_ids_pos, next_token_scores_pos)
                if input_ids_neg is not None: next_token_scores_neg = logits_warper(input_ids_neg, next_token_scores_neg)
            
            if input_ids_pos is None and input_ids_neg is None: # mode == 'naive'
                # Naive decoding
                next_token_scores = next_token_scores
            elif input_ids_pos is not None and mode == 'acd':
                # Our proposed decoding
                probs = F.softmax(next_token_scores, dim=-1)
                probs_pos = F.softmax(next_token_scores_pos, dim=-1)
                
                # shannons entropy with base 2
                ent = -torch.sum(probs * torch.log2(probs), dim=-1)
                ent_pos = -torch.sum(probs_pos * torch.log2(probs_pos), dim=-1)
                
                d_alpha = ((ent)/(ent_pos + ent)).unsqueeze(-1)
                
                next_token_scores = next_token_scores + d_alpha * (next_token_scores_pos - next_token_scores)
            
            else:
                raise ValueError(f"Invalid mode: {mode}")

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.gen_model.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.gen_model.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.gen_model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = F.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if input_ids_pos is not None: input_ids_pos = torch.cat([input_ids_pos, next_tokens[:, None]], dim=-1)
            if input_ids_neg is not None: input_ids_neg = torch.cat([input_ids_neg, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self.gen_model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.gen_model.config.is_encoder_decoder,
            )
            if input_ids_pos is not None:
                model_kwargs_pos = self.gen_model._update_model_kwargs_for_generation(
                    outputs_pos,
                    model_kwargs_pos,
                    is_encoder_decoder=self.gen_model.config.is_encoder_decoder,
                )
            if input_ids_neg is not None:
                model_kwargs_neg = self.gen_model._update_model_kwargs_for_generation(
                    outputs_neg,
                    model_kwargs_neg,
                    is_encoder_decoder=self.gen_model.config.is_encoder_decoder,
                )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs
            if input_ids_pos is not None: del outputs_pos
            if input_ids_neg is not None: del outputs_neg

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.gen_model.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
        