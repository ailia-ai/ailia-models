export OPTION="-e 1"
cd ../
cd audio_processing/clap/; python3 clap.py ${OPTION}
cd ../../audio_processing/distil-whisper/; python3 distil-whisper.py ${OPTION}
cd ../../audio_processing/msclap/; python3 msclap.py -v 2022 ${OPTION}
cd ../../audio_processing/msclap/; python3 msclap.py -v 2023 ${OPTION}
cd ../../audio_processing/kotoba-whisper/; python3 kotoba-whisper.py ${OPTION}
cd ../../diffusion/latent-diffusion-txt2img; python3 latent-diffusion-txt2img.py ${OPTION}
cd ../../diffusion/stable-diffusion-txt2img; python3 stable-diffusion-txt2img.py ${OPTION}
cd ../../diffusion/control_net python3 control_net.py ${OPTION}
cd ../../diffusion/riffusion; python3 riffusion.py ${OPTION}
cd ../../diffusion/marigold; python3 marigold.py ${OPTION}
cd ../../image_captioning/blip2; python3 blip2.py ${OPTION}
cd ../../image_classification/japanese-clip; python3 japanese-clip.py ${OPTION}
cd ../../image_classification/siglip2; python3 siglip2.py ${OPTION}
cd ../../large_language_model/llava; python3 llava.py ${OPTION}
cd ../../natural_language_processing/bert; python3 bert.py ${OPTION}
cd ../../natural_language_processing/bert_insert_punctuation; python3 bert_insert_punctuation.py ${OPTION}
cd ../../natural_language_processing/bert_maskedlm; python3 bert_maskedlm.py -a bert-base-cased ${OPTION}
cd ../../natural_language_processing/bert_maskedlm; python3 bert_maskedlm.py -a bert-base-uncased ${OPTION}
cd ../../natural_language_processing/bert_maskedlm; python3 bert_maskedlm.py -a bert-base-japanese-whole-word-masking ${OPTION}
cd ../../natural_language_processing/bert_maskedlm; python3 bert_maskedlm.py -a bert-base-japanese-char-whole-word-masking ${OPTION}
cd ../../natural_language_processing/bert_maskedlm; python3 bert_maskedlm.py -a bert-base-japanese-v3 ${OPTION}
cd ../../natural_language_processing/bert_maskedlm; python3 bert_maskedlm.py -a bert-base-japanese-char-v3 ${OPTION}
cd ../../natural_language_processing/bert_maskedlm; python3 bert_maskedlm.py ${OPTION}
cd ../../natural_language_processing/bert_maskedlm; python3 bert_maskedlm_proofreeding.py -a bert-base-cased ${OPTION}
cd ../../natural_language_processing/bert_maskedlm; python3 bert_maskedlm_proofreeding.py -a bert-base-uncased ${OPTION}
cd ../../natural_language_processing/bert_maskedlm; python3 bert_maskedlm_proofreeding.py -a bert-base-japanese-whole-word-masking ${OPTION}
cd ../../natural_language_processing/bert_ner; python3 bert_ner.py ${OPTION}
cd ../../natural_language_processing/bert_question_answering; python3 bert_question_answering.py ${OPTION}
cd ../../natural_language_processing/bert_sentiment_analysis; python3 bert_sentiment_analysis.py ${OPTION}
cd ../../natural_language_processing/bert_tweets_sentiment; python3 bert_tweets_sentiment.py ${OPTION}
cd ../../natural_language_processing/bert_zero_shot_classification; python3 bert_zero_shot_classification.py ${OPTION}
cd ../../natural_language_processing/bertjsc; python3 bertjsc.py ${OPTION}
cd ../../natural_language_processing/gpt2; python3 gpt2.py ${OPTION}
cd ../../natural_language_processing/rinna_gpt2; python3 rinna_gpt2.py ${OPTION}
cd ../../natural_language_processing/fugumt-en-ja; python3 fugumt-en-ja.py ${OPTION}
cd ../../natural_language_processing/fugumt-ja-en; python3 fugumt-en-ja.py ${OPTION}
cd ../../natural_language_processing/bert_sum_ext; python3 bert_sum_ext.py ${OPTION}
cd ../../natural_language_processing/sentence_transformers_japanese; python3 sentence_transformers_japanese.py -p "what is ailia SDK?" ${OPTION}
cd ../../natural_language_processing/t5_base_japanese_title_generation; python3 t5_base_japanese_title_generation.py ${OPTION}
cd ../../natural_language_processing/multilingual-e5; python3 multilingual-e5.py -p "what is ailia SDK?" ${OPTION}
cd ../../natural_language_processing/t5_whisper_medical; python3 t5_whisper_medical.py ${OPTION}
cd ../../natural_language_processing/t5_base_japanese_summarization; python3 t5_base_japanese_summarization.py ${OPTION}
cd ../../natural_language_processing/glucose; python3 glucose.py ${OPTION}
cd ../../natural_language_processing/cross_encoder_mmarco; python3 cross_encoder_mmarco.py ${OPTION}
cd ../../natural_language_processing/soundchoice-g2p; python3 soundchoice-g2p.py ${OPTION}
cd ../../natural_language_processing/multilingual-minilmv2; python3 multilingual-minilmv2.py ${OPTION}
cd ../../natural_language_processing/embeddinggemma; python3 embeddinggemma.py ${OPTION}
cd ../../network_intrusion_detection/bert-network-packet-flow-header-payload; python3 bert-network-packet-flow-header-payload.py ${OPTION}
cd ../../network_intrusion_detection/falcon-adapter-network-packet; python3 falcon-adapter-network-packet.py ${OPTION}
cd ../../object_detection/glip; python3 glip.py ${OPTION}
