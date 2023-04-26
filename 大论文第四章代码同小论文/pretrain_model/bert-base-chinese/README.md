---
language: zh
---




- [Model Details](
- [Uses](
- [Risks, Limitations and Biases](
- [Training](
- [Evaluation](
- [How to Get Started With the Model](



- **Model Description:**
This model has been pre-trained for Chinese, training and random input masking has been applied independently to word pieces (as in the original BERT paper).

- **Developed by:** HuggingFace team
- **Model Type:** Fill-Mask
- **Language(s):** Chinese
- **License:** [More Information needed]
- **Parent Model:** See the [BERT base uncased model](https://huggingface.co/bert-base-uncased) for more information about the BERT base model.






This model can be used for masked language modeling 




**CONTENT WARNING: Readers should be aware this section contains content that is disturbing, offensive, and can propagate historical and current stereotypes.**

Significant research has explored bias and fairness issues with language models (see, e.g., [Sheng et al. (2021)](https://aclanthology.org/2021.acl-long.330.pdf) and [Bender et al. (2021)](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)).





* **type_vocab_size:** 2
* **vocab_size:** 21128
* **num_hidden_layers:** 12


[More Information Needed]





[More Information Needed]



```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")

```





