# FP32のSenseVoiceをFP16に変換する

# onnxconverter-common : 1.16.0
# onnx : 1.18.0

import onnx
from onnxconverter_common.float16 import convert_float_to_float16

ONNXRUNTIME_SUPPORT = False

if ONNXRUNTIME_SUPPORT:
	op_block_list = [
		"ArrayFeatureExtractor",
		"Binarizer",
		"CastMap",
		"CategoryMapper",
		"DictVectorizer",
		"FeatureVectorizer",
		"Imputer",
		"LabelEncoder",
		"LinearClassifier",
		"LinearRegressor",
		"Normalizer",
		"OneHotEncoder",
		"RandomUniformLike",
		"SVMClassifier",
		"SVMRegressor",
		"Scaler",
		"TreeEnsembleClassifier",
		"TreeEnsembleRegressor",
		"ZipMap",
		"NonMaxSuppression",
		"TopK",
		"RoiAlign",
		"Resize",
		"Range",
		"CumSum",
		"Min",
		"Max",
		"Upsample",
		"Range", "Equal", "Where" # ONNX Runtimeではこの3つのオペレータはFP16に対応していない
	]
else:
	op_block_list = None

model = onnx.load("../sensevoice_small.onnx")
model_fp16 = convert_float_to_float16(model, disable_shape_infer=False, keep_io_types=False, op_block_list = op_block_list)
onnx.save(model_fp16, "../sensevoice_small_fp16.onnx.tmp")

model = onnx.load("../speech_fsmn_vad_zh-cn-16k-common.onnx")
model_fp16 = convert_float_to_float16(model, disable_shape_infer=False, keep_io_types=False, op_block_list = op_block_list)
onnx.save(model_fp16, "../speech_fsmn_vad_zh-cn-16k-common_fp16.onnx")

import onnx
from onnx import helper, numpy_helper
import numpy as np

def modify_onnx(model_path: str, output_path: str):
	model = onnx.load(model_path)
	graph = model.graph
	modified = False

	# 元のグラフに含まれていたFloat32へのCastをFloat16へのCastに置き換え
	for node in graph.node:
		if node.op_type != "Cast":
			continue

		# Range、Equal、WWhere に接続されている場合はスキップ
		cast_output = node.output[0] if len(node.output) > 0 else None
		if not cast_output:
			continue

		connected_to_range = False
		for other in graph.node:
			if other.op_type == "Range" or other.op_type == "Equal" or other.op_type == "Where":
				if cast_output in other.input:
					connected_to_range = True
					break

		if connected_to_range and ONNXRUNTIME_SUPPORT:
			continue

		for attr in node.attribute:
			if attr.name == "to" and attr.i == 1:  # float32
				print(f"Changing Cast('{node.name}') from float32 (1) to float16 (10)")
				attr.i = 10  # float16
				modified = True

	# SubのInitilaizerをFloat16に置き換え
	initializer_dict = {init.name: init for init in graph.initializer}

	for node in graph.node:
		if node.op_type != "Sub" and (ONNXRUNTIME_SUPPORT or node.op_type != "Equal"):
			continue

		for inp_name in node.input:
			if inp_name not in initializer_dict:
				continue

			init_tensor = initializer_dict[inp_name]

			if init_tensor.data_type == 1:  # float32
				np_data = numpy_helper.to_array(init_tensor).astype(np.float16)
				new_init = numpy_helper.from_array(np_data, name=init_tensor.name)

				# 古いinitializerを置き換え
				index = list(graph.initializer).index(init_tensor)
				graph.initializer.remove(init_tensor)
				graph.initializer.insert(index, new_init)

				print(f"Converted initializer '{inp_name}' ({node.op_type} input) from float32 to float16")
				modified = True

	# RangeのInitializerをFloat32に置き換え
	initializer_dict = {init.name: init for init in graph.initializer}

	if ONNXRUNTIME_SUPPORT:
		for node in graph.node:
			if node.op_type != "Range":
				continue

			for i, inp_name in enumerate(node.input):
				if inp_name not in initializer_dict:
					continue

				init_tensor = initializer_dict[inp_name]

				if init_tensor.data_type == 10:  # float16
					np_data = numpy_helper.to_array(init_tensor).astype(np.float32)
					new_name = init_tensor.name+"_fp32"
					new_init = numpy_helper.from_array(np_data, name=new_name)

					# 新しいinitializerを追加
					# 1つのinitializerが複数参照されているので削除はしない
					graph.initializer.append(new_init)
					node.input[i] = new_name

					print(f"Converted initializer '{inp_name}' (Range input) from float16 to float32")
					modified = True

	onnx.save(model, output_path)
	print(f"Saved modified model to: {output_path}")

	if modified:
		print("Modifications were applied.")
	else:
		print("No modifications were applied.")

modify_onnx("../sensevoice_small_fp16.onnx.tmp", "../sensevoice_small_fp16.onnx")

import subprocess
subprocess.call("python3 onnx2prototxt.py " + "../sensevoice_small_fp16.onnx", shell=True)
subprocess.call("python3 onnx2prototxt.py " + "../speech_fsmn_vad_zh-cn-16k-common_fp16.onnx", shell=True)