# Authors: Nicolas Legrand <nicolas.legrand@cas.au.dk> & Aleksandrs Baskakovs <aleks@mgmt.au.dk>

import torch

from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import List, Tuple

class ThoughtMiner:
    """
    Utility class to extract hidden state trajectories from language model outputs
    generated during reasoning tasks.

    Args:
        model (PreTrainedModel): The pretrained HuggingFace model for generation.
        tokenizer (PreTrainedTokenizerBase): The corresponding tokenizer.
        max_length (int, optional): Maximum sequence length for generation. Default is 2048.
        top_k (int, optional): Top-k sampling for generation (higher = more diverse). Default is 50.
        temperature (float, optional): Softmax temperature for generation (higher = more random). Default is 0.8.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        top_k: int = 50,
        temperature: float = 0.8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length            # Controls how many tokens the model generates
        self.top_k = top_k                      # Limits sampling to top_k most likely tokens
        self.temperature = temperature          # Controls randomness; higher = more diverse output

    def forward_pass(
        self, inputs, num_return_sequences: int = 1
    ) -> GenerateDecoderOnlyOutput:
        """
        Generates model outputs from inputs with hidden states.
        """
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=self.max_length,
            do_sample=True,
            top_k=self.top_k,
            temperature=self.temperature,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        return outputs
    
    def forward_pass_unsloth(
        self, inputs, num_return_sequences: int = 1
    ) -> GenerateDecoderOnlyOutput:
        """
        Performs a forward pass without sampling-based generation.
        Suitable for compatibility with Unsloth training loop.

        Returns:
            GenerateDecoderOnlyOutput: Includes logits and hidden states.
        """
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
        return outputs

    def get_continuous_thoughts(
        self,
        outputs: GenerateDecoderOnlyOutput,
        input_length: int,
        line_delimiter: str = "\n\n",
    ) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        
        """
        Splits every generated response on double–line breaks, and returns
        per‐step hidden‐state sequences + token‐ID spans.
        Works purely on outputs from `.generate(return_dict_in_generate=True)`.
        """
    
        # 1) drop the prompt tokens
        responses = outputs.sequences[:, input_length:]              # (batch, gen_len)
    
        # 2) decode each response to text
        decoded = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in responses
        ]
    
        # 3) grab the last layer hidden‐states from generate()
        hs_layers = outputs.hidden_states
        # we want hs_layers[t][-1] → tensor(batch, hidden_dim)
        # build a single Tensor (batch, gen_len, hidden_dim)
        last_layer = torch.stack(
            [ layer_out[-1] for layer_out in hs_layers[input_length:] ],
            dim=1
        )
        # now last_layer[b, t, :] is the vector for response token t of batch b
    
        all_trajs = []
        all_toks  = []
        for b, text in enumerate(decoded):
            steps    = [s for s in text.split(line_delimiter) if s.strip()]
            tokspan  = responses[b]
            ptr      = 0
            trajs, toks = [], []
    
            for step in steps:
                # re-tokenize the slice to know its length
                step_ids = self.tokenizer(
                    step,
                    add_special_tokens=False
                ).input_ids
                L = len(step_ids)
                if ptr + L > tokspan.size(0):
                    break
                # slice out hidden-states & token-ids
                trajs.append( last_layer[b, ptr:ptr+L] )
                toks.append(  tokspan[ ptr:ptr+L]       )
                ptr += L
    
            all_trajs.append(trajs)
            all_toks.append(toks)
    
        return all_trajs, all_toks
    
    def get_continuous_thoughts_unsloth(
            self,
            inputs,  # we now need inputs as well
            outputs,
            line_delimiter: str = "\n\n",
        ) -> tuple[list, list]:
            """
            Splits output into reasoning steps using line breaks.
            Works with model() forward pass, not generate().
            """

            input_ids = inputs["input_ids"]
            batch_size = input_ids.shape[0]

            # Decode each full input to match with the hidden states
            decoded_texts = [self.tokenizer.decode(x, skip_special_tokens=True) for x in input_ids]

            # Last hidden layer (batch, seq, dim)
            hidden_states = outputs.hidden_states[-1]

            continuous_trajectories, token_trajectories = [], []

            for batch_idx in range(batch_size):
                text = decoded_texts[batch_idx]
                tokens = input_ids[batch_idx]

                # Split reasoning steps using double line breaks
                #lines = [line.strip() for line in text.split(line_delimiter) if line.strip()]

                #lines = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

                lines = [chunk.strip()
                        for chunk in re.split(r'\n\s*\n', text)
                        if chunk.strip()]

                # Tokenize each line separately
                step_token_ids = [self.tokenizer(line, return_tensors="pt", add_special_tokens=False).input_ids[0]
                                for line in lines]

                pointer = 0
                traj, tok = [], []
                for step_tokens in step_token_ids:
                    length = len(step_tokens)
                    if pointer + length > tokens.size(0):
                        break  # overflow guard
                    traj.append(hidden_states[batch_idx, pointer:pointer+length])
                    tok.append(tokens[pointer:pointer+length])
                    pointer += length

                continuous_trajectories.append(traj)
                token_trajectories.append(tok)

            return continuous_trajectories, token_trajectories