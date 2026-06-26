# Donut: Document Understanding Transformer

## Input

![input image](cord_sample_receipt1.png)

(from https://huggingface.co/spaces/naver-clova-ix/donut-base-finetuned-cord-v2)

<br/>

## Output

```json
{
    "menu": [
        {
            "nm": "ICE BLAOKCOFFE",
            "cnt": "2",
            "price": "82,000"
        },
        {
            "nm": "AVOCADO COFFEE",
            "cnt": "1",
            "price": "61,000"
        },
        {
            "nm": "Oud CHINEN KATSU FF",
            "cnt": "1",
            "price": "51,000"
        }
    ],
    "sub_total": {
        "subtotal_price": "194,000",
        "discount_price": "19,400"
    },
    "total": {
        "total_price": "174,600",
        "cashprice": "200,000",
        "changeprice": "25,400"
    }
}
```

## Requirements

This model requires additional module.

```
pip3 install transformers
```

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 donut.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 donut.py --input IMAGE_PATH
```

By adding the `--model_name` option, you can specify model name. (default is 'donut-base-finetuned-cord-v2')
```bash
$ python3 donut.py --model_name donut-base-finetuned-cord-v2
```

## Reference

- [Donut](https://github.com/clovaai/donut)

## Model export

Follow the steps below to export your custom model.

1. Clone the original donut repository
   ```
   git clone https://github.com/clovaai/donut
   ```

2. Install the required modules.
   ```
   pip3 install torch torchvision
   pip3 install transformers==4.48.1
   pip3 install timm==0.5.4
   ```

3. Run the command specifying your custom model.
   ```
   python3 export_model.py --pretrained_path naver-clova-ix/donut-base-finetuned-cord-v2 --half
   ```

4. An export file for the specified model will be created.
   ```
   donut-base-finetuned-cord-v2_encoder.onnx created.
   donut-base-finetuned-cord-v2.onnx created.
   ```

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[donut-base-finetuned-cord-v2_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/donut/donut-base-finetuned-cord-v2_encoder.onnx.prototxt)  
[donut-base-finetuned-cord-v2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/donut/donut-base-finetuned-cord-v2.onnx.prototxt)  
