# ailia onnx to prototxt
# (c) 2020-2022 AXELL CORPORATION

import sys
import onnx
import json


def dump_normal(elem, indent, file):
    # torch.onnx (dynamo) attaches metadata_props to nodes and value infos;
    # ailia's prototxt parser rejects the keyword, so they are dropped
    if hasattr(elem, "metadata_props"):
        del elem.metadata_props[:]
    for s in str(elem).splitlines():
        print(indent + s, file=file)


def dump_initializer(elem, indent, file):
    # output metadata
    for d in elem.dims:
        print(indent + "  dims: " + json.dumps(d), file=file)
    print(indent + "  data_type: " + json.dumps(elem.data_type), file=file)
    print(indent + "  name: " + json.dumps(elem.name), file=file)


def dump_constant(elem, indent, file):
    # print output & name & op_type
    print(indent + "output: " + json.dumps(elem.output[0]), file=file)
    if len(elem.name) > 0:
        print(indent + "name: " + json.dumps(elem.name), file=file)
    print(indent + "op_type: " + json.dumps(elem.op_type), file=file)

    for a in elem.attribute:
        print(indent + "attribute {", file=file)
        if (a.name == "value") and (a.type == onnx.AttributeProto.TENSOR):
            print(indent + "  name: " + json.dumps(a.name), file=file)
            print(indent + "  t {", file=file)
            for d in a.t.dims:
                print(indent + "    dims: " + json.dumps(d), file=file)
            print(indent + "    data_type: " + json.dumps(a.t.data_type), file=file)
            if hasattr(a.t, 'data_location'):
                print(indent + "    data_location: " + json.dumps(a.t.data_location), file=file)
            print(indent + "  }", file=file)
            print(indent + "  type: TENSOR", file=file)
        else:
            dump_normal(a, indent + "  ", file)
        print(indent + "}", file=file)


def onnx2prototxt(onnx_path):

    # show information
    out_path = onnx_path + ".prototxt"
    print("+ creating " + out_path)
    print("    from " + onnx_path + " ...")

    # load model
    model = onnx.load(onnx_path, load_external_data=False)

    # print prototxt
    with open(out_path, "w") as f:
        print("ir_version: " + json.dumps(model.ir_version), file=f)
        print("producer_name: " + json.dumps(model.producer_name), file=f)
        print("producer_version: " + json.dumps(model.producer_version), file=f)
        # print("domain: " + json.dumps(model.domain), file=f)
        print("model_version: " + json.dumps(model.model_version), file=f)
        # print("doc_string: " + json.dumps(model.doc_string), file=f)
        print("graph {", file=f)
        print("  name: " + json.dumps(model.graph.name), file=f)

        for e in model.graph.node:
            print("  node {", file=f)
            if e.op_type == "Constant":
                dump_constant(e, "    ", f)
            else:
                dump_normal(e, "    ", f)
            print("  }", file=f)

        for e in model.graph.initializer:
            print("  initializer {", file=f)
            dump_initializer(e, "    ", f)
            print("  }", file=f)

        for e in model.graph.input:
            print("  input {", file=f)
            dump_normal(e, "    ", f)
            print("  }", file=f)

        for e in model.graph.output:
            print("  output {", file=f)
            dump_normal(e, "    ", f)
            print("  }", file=f)

        print("}", file=f)

        for e in model.opset_import:
            print("opset_import {", file=f)
            print("  domain: " + json.dumps(e.domain), file=f)
            print("  version: " + json.dumps(e.version), file=f)
            print("}", file=f)


def show_usage(script):
    print("usage: python " + script + " input.onnx [more.onnx ..]")


def main():
    if len(sys.argv) == 1:
        show_usage(sys.argv[0])
        return

    for i in range(1, len(sys.argv)):
        onnx2prototxt(sys.argv[i])


if __name__ == "__main__":
    main()
