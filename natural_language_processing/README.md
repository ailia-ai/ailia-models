[ailia MODELS](../README.md) > Natural language processing

# ailia MODELS : Natural language processing

### Bert

| Model | Reference | Exported From | Supported Ailia Version | Date | Blog |
|------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
|[bert](./bert) | [pytorch-pretrained-bert](https://pypi.org/project/pytorch-pretrained-bert/) | Pytorch | 1.2.2 and later | Oct 2018 | [EN](https://tech.ailia.ai/en/bert-a-machine-learning-model-for-efficient-natural-language-processing-aef3081c24e8) [JP](https://tech.ailia.ai/bert-%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86%E3%82%92%E5%8A%B9%E7%8E%87%E7%9A%84%E3%81%AB%E5%AD%A6%E7%BF%92%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-3a9c27d78cf8) |
|[bert_maskedlm](./bert_maskedlm) | [huggingface/transformers](https://github.com/huggingface/transformers) | Pytorch | 1.2.5 and later | Oct 2018 | |
|[bert_question_answering](./bert_question_answering) | [huggingface/transformers](https://github.com/huggingface/transformers) | Pytorch | 1.2.5 and later | Oct 2018 | |

### Embedding

| Model | Reference | Exported From | Supported Ailia Version | Date | Blog |
|------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
|[sentence_transformers_japanese](./sentence_transformers_japanese) | [sentence transformers](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) | Pytorch | 1.2.7 and later | Aug 2019 | [JP](https://tech.ailia.ai/sentencetransformer-%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E3%81%8B%E3%82%89embedding%E3%82%92%E5%8F%96%E5%BE%97%E3%81%99%E3%82%8B%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86%E3%83%A2%E3%83%87%E3%83%AB-b7d2a9bb2c31) |
|[multilingual-e5](./multilingual-e5) | [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) | Pytorch | 1.2.15 and later | Dec 2022 | [EN](https://tech.ailia.ai/en/multilingual-e5-a-machine-learning-model-for-embedding-text-in-multiple-languages-b4916cb22bda/) [JP](https://tech.ailia.ai/multilingual-e5-%E5%A4%9A%E8%A8%80%E8%AA%9E%E3%81%AE%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E3%82%92embedding%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-71f1dec7c4f0) |
|[glucose](./glucose) | [GLuCoSE (General Luke-based Contrastive Sentence Embedding)-base-Japanese](https://huggingface.co/pkshatech/GLuCoSE-base-ja) | Pytorch | 1.2.15 and later | Jul 2023 | |
|[qwen3-embedding](./qwen3-embedding) | [Hugging Face - Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Pytorch | 1.2.16 and later | Jun 2025| |
|[ruri-v3](./ruri-v3) | [ruri-v3-310m ](https://huggingface.co/cl-nagoya/ruri-v3-310m) | Pytorch | 1.2.13 and later | Apr 2025 | |
|[embeddinggemma](./embeddinggemma) | [EmbeddingGemma](https://ai.google.dev/gemma/docs/embeddinggemma?hl=ja) | Pytorch | 1.2.14 and later | Sep 2025| [JP](https://kyakuno.medium.com/embedding-gemma-google%E3%81%AE%E9%96%8B%E7%99%BA%E3%81%97%E3%81%9F%E8%BB%BD%E9%87%8F%E3%81%A7%E9%AB%98%E7%B2%BE%E5%BA%A6%E3%81%AAembedding%E3%83%A2%E3%83%87%E3%83%AB-9ec139ddfde9) |

### Error corrector

| Model | Reference | Exported From | Supported Ailia Version | Date | Blog |
|------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
|[bert_insert_punctuation](./bert_insert_punctuation) | [bert-japanese](https://github.com/cl-tohoku/bert-japanese) | Pytorch | 1.2.15 and later | Nov 2019 | |
|[bertjsc](./bertjsc) | [bertjsc](https://github.com/er-ri/bertjsc) | Pytorch | 1.2.15 and later | Mar 2023 | |
|[t5_whisper_medical](./t5_whisper_medical) | error correction of medical terms using t5 | Pytorch | 1.2.13 and later |  | |

### Grapheme to phoneme

| Model | Reference | Exported From | Supported Ailia Version | Date | Blog |
|------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
|[g2p_en](./g2p_en) | [g2p_en](https://github.com/Kyubyong/g2p) | Pytorch | 1.2.14 and later | Jan 2019 | [EN](https://tech.ailia.ai/en/g2p-en-a-machine-learning-model-for-converting-english-text-to-phonemes-03072cc2251f/) [JP](https://tech.ailia.ai/g2p-en-%E8%8B%B1%E8%AA%9E%E3%81%AE%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E3%82%92%E9%9F%B3%E7%B4%A0%E3%81%AB%E5%A4%89%E6%8F%9B%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-88947c27b9ea) |
|[g2pw](./g2pw) | [g2pW](https://github.com/GitYCC/g2pW) | Pytorch | 1.2.9 and later | Mar 2022 | |
|[soundchoice-g2p](./soundchoice-g2p) | [Hugging Face - speechbrain/soundchoice-g2p](https://huggingface.co/speechbrain/soundchoice-g2p) | Pytorch | 1.2.16 and later | Jul 2022 | |

### Named entity recognition

| Model | Reference | Exported From | Supported Ailia Version | Date | Blog |
|------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
|[bert_ner](./bert_ner) | [huggingface/transformers](https://github.com/huggingface/transformers) | Pytorch | 1.2.5 and later | Oct 2018 | |
|[t5_base_japanese_ner](./t5_base_japanese_ner) |  [t5-japanese](https://github.com/sonoisa/t5-japanese) | Pytorch | 1.2.13 and later | Mar 2021 | |
|[bert_ner_japanese](./bert_ner_japanese) | [jurabi/bert-ner-japanese](https://huggingface.co/jurabi/bert-ner-japanese) | Pytorch | 1.2.10 and later | Mar 2023 | |

### Reranker

| Model | Reference | Exported From | Supported Ailia Version | Date | Blog |
|------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
|[cross_encoder_mmarco](./cross_encoder_mmarco) | [jeffwan/mmarco-mMiniLMv2-L12-H384-v](https://huggingface.co/jeffwan/mmarco-mMiniLMv2-L12-H384-v1) | Pytorch | 1.2.10 and later | Sep 2022 | [EN](https://tech.ailia.ai/en/crossencodermmarco-machine-learning-model-that-calculates-the-similarity-between-a-question-and-an-1906f716f8ef/) [JP](https://tech.ailia.ai/crossencodermmarco-%E8%B3%AA%E5%95%8F%E6%96%87%E3%81%A8%E5%9B%9E%E7%AD%94%E6%96%87%E3%81%AE%E9%A1%9E%E4%BC%BC%E5%BA%A6%E3%82%92%E8%A8%88%E7%AE%97%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-c90b35e9fc09)|
|[japanese-reranker-cross-encoder](./japanese-reranker-cross-encoder) | [hotchpotch/japanese-reranker-cross-encoder-large-v1](https://huggingface.co/hotchpotch/japanese-reranker-cross-encoder-large-v1) | Pytorch | 1.2.16 and later | Apr 2024 | |
|[ruri-v3-reranker](./ruri-v3-reranker) | [ruri-v3-reranker-310m ](https://huggingface.co/cl-nagoya/ruri-v3-reranker-310m) | Pytorch | 1.2.16 and later | Apr 2025 | |

### Sentence generation

| Model | Reference | Exported From | Supported Ailia Version | Date | Blog |
|------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
|[gpt2](./gpt2) | [GPT-2](https://github.com/onnx/models/blob/master/text/machine_comprehension/gpt-2/README.md) | Pytorch | 1.2.7 and later | Feb 2019 | |
|[rinna_gpt2](./rinna_gpt2) | [japanese-pretrained-models](https://github.com/rinnakk/japanese-pretrained-models)   | Pytorch | 1.2.7 and later | Apr 2021 | |

### Sentiment analysis

| Model | Reference | Exported From | Supported Ailia Version | Date | Blog |
|------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
|[bert_sentiment_analysis](./bert_sentiment_analysis) | [huggingface/transformers](https://github.com/huggingface/transformers) | Pytorch | 1.2.5 and later | Oct 2018 | |
|[bert_tweets_sentiment](./bert_tweets_sentiment) | [huggingface/transformers](https://github.com/huggingface/transformers) | Pytorch | 1.2.5 and later | Oct 2018 | |

### Summarize

| Model | Reference | Exported From | Supported Ailia Version | Date | Blog |
|------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
|[bert_sum_ext](./bert_sum_ext) | [BERTSUMEXT](https://github.com/dmmiller612/bert-extractive-summarizer)   | Pytorch | 1.2.7 and later | May 2019 | |
|[presumm](./presumm) | [PreSumm](https://github.com/nlpyang/PreSumm)   | Pytorch | 1.2.8 and later| Aug 2019 | |
|[t5_base_japanese_title_generation](./t5_base_japanese_title_generation) | [t5-japanese](https://github.com/sonoisa/t5-japanese) | Pytorch | 1.2.13 and later | Mar 2021 | [JP](https://tech.ailia.ai/t5-%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E3%81%8B%E3%82%89%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E3%82%92%E7%94%9F%E6%88%90%E3%81%99%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-602830bdc5b4) |
|[t5_base_summarization](./t5_base_japanese_summarization) | [t5-japanese](https://github.com/sonoisa/t5-japanese) | Pytorch | 1.2.13 and later | Mar 2021 | |

### Translation

| Model | Reference | Exported From | Supported Ailia Version | Date | Blog |
|------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
|[fugumt-en-ja](./fugumt-en-ja) | [Fugu-Machine Translator](https://github.com/s-taka/fugumt)   | Pytorch | 1.2.9 and later | Nov 2020 | [JP](https://tech.ailia.ai/fugumt-%E8%8B%B1%E8%AA%9E%E3%81%8B%E3%82%89%E6%97%A5%E6%9C%AC%E8%AA%9E%E3%81%B8%E3%81%AE%E7%BF%BB%E8%A8%B3%E3%82%92%E8%A1%8C%E3%81%86%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB-46b839c1b4ae) |
|[fugumt-ja-en](./fugumt-ja-en) | [Fugu-Machine Translator](https://github.com/s-taka/fugumt)   | Pytorch | 1.2.10 abd later | Nov 2020 | |

### Zero shot classification

| Model | Reference | Exported From | Supported Ailia Version | Date | Blog |
|------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
|[bert_zero_shot_classification](./bert_zero_shot_classification) | [huggingface/transformers](https://github.com/huggingface/transformers) | Pytorch | 1.2.5 and later | Oct 2018 | |
|[multilingual-minilmv2](./multilingual-minilmv2) | [MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli](https://huggingface.co/MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli) | Pytorch | 1.2.10 and later | Jun 2022 | |

[Back to the model list](../README.md)
