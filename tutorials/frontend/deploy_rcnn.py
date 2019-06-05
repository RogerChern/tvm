import mxnet as mx
from tvm import relay
from tvm.relay.testing.config import ctx_list

sym = mx.sym.load("/mnt/diskc/yuntao_chen/checkpoint_test.json")
params = mx.nd.load("/mnt/diskc/yuntao_chen/checkpoint-0006.params")
arg, aux = {}, {}
for k in params.keys():
    if k.startswith("arg:"):
        arg[k[4:]] = params[k]
    elif k.startswith("aux:"):
        aux[k[4:]] = params[k]
net, params = relay.frontend.from_mxnet(sym, {"data": (1, 3, 200, 300), "im_info": (1, 3), "rec_id": (1, ), "im_id": (1, )}, arg_params=arg, aux_params=aux)

with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(net, "cuda", params=params)

with open("rcnn.json", "w") as fout:
    fout.write(graph)
