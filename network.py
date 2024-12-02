import torch
from typing import List, NamedTuple
from math import exp


class NetworkOutput(NamedTuple):
    value: float
    reward: float
    policy_logits: List[float]


class PerplexityCalculator:
    def __init__(self, model, tokenizer, device, prompt=""):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.prompt = prompt
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    def get_perplexity(self, input_texts):
        single_input = isinstance(input_texts, str)
        input_texts = [input_texts] if single_input else input_texts
        input_texts = [self.prompt + t for t in input_texts]
        loss_list = []
        policy_logits = []
        with torch.no_grad():
            for text in input_texts:
                text_with_special = f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}"
                minputs = self.tokenizer(text_with_special, return_tensors='pt', add_special_tokens=False, )
                if 'token_type_ids' in minputs:
                    minputs.pop('token_type_ids')
                minputs = {k: v.to(self.device) for k, v in minputs.items()}
                output = self.model(**minputs, use_cache=False)
                logits = output['logits']
                slogits = logits[..., :-1, :].contiguous()
                slabels = minputs['input_ids'][..., 1:].contiguous()
                loss = self.loss_fct(slogits.view(-1, slogits.size(-1)), slabels.view(-1))
                sequence_loss = loss.sum() / len(loss)
                loss_list.append(sequence_loss.cpu().item())

                policy_logits.append(slogits[0, -1].detach().cpu().numpy())

        perplexity = [exp(i) for i in loss_list]
        if single_input:
            perplexity, policy_logits = perplexity.pop(), policy_logits.pop()
        return perplexity, policy_logits


class Network(object):
    def __init__(self, scorer: PerplexityCalculator):
        self.scorer = scorer

    def get_token_id(self, token):
        return self.scorer.tokenizer.encode(token, add_special_tokens=False)[0]

    def inference(self, text) -> NetworkOutput:
        perplexity, logits = self.scorer.get_perplexity(text)
        value = -perplexity
        return NetworkOutput(value, 0, logits)
