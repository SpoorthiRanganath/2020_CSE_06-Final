from utils.google_utils import *
from utils.layers import *
from utils.parse_config import *

ONNX_EXPORT = False


def create_modules(module_defs, img_size):

    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
    _ = module_defs.pop(0)
    output_filters = [3]
    module_list = nn.ModuleList()
    routs = []
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            size = mdef['size']
            stride = mdef['stride'] if 'stride' in mdef else (
                mdef['stride_y'], mdef['stride_x'])
            if isinstance(size, int):
                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=size,
                                                       stride=stride,
                                                       padding=(
                                                           size - 1) // 2 if mdef['pad'] else 0,
                                                       groups=mdef['groups'] if 'groups' in mdef else 1,
                                                       bias=not bn))
            else:  # multiple-size conv
                modules.add_module('MixConv2d', MixConv2d(in_ch=output_filters[-1],
                                                          out_ch=filters,
                                                          k=size,
                                                          stride=stride,
                                                          bias=not bn))

            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(
                    filters, momentum=0.03, eps=1E-4))
            else:
                routs.append(i)

            if mdef['activation'] == 'leaky':
                modules.add_module(
                    'activation', nn.LeakyReLU(0.1, inplace=True))

            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())

        elif mdef['type'] == 'BatchNorm2d':
            filters = output_filters[-1]
            modules = nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4)
            if i == 0 and filters == 3:

                modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
                modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])

        elif mdef['type'] == 'maxpool':
            size = mdef['size']
            stride = mdef['stride']
            maxpool = nn.MaxPool2d(
                kernel_size=size, stride=stride, padding=(size - 1) // 2)
            if size == 2 and stride == 1:
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        elif mdef['type'] == 'upsample':
            if ONNX_EXPORT:
                g = (yolo_index + 1) * 2 / 32
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))
            else:
                modules = nn.Upsample(scale_factor=mdef['stride'])

        elif mdef['type'] == 'route':
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l]
                          for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)

        elif mdef['type'] == 'shortcut':
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = WeightedFeatureFusion(
                layers=layers, weight='weights_type' in mdef)

        elif mdef['type'] == 'reorg3d':
            pass

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            stride = [32, 16, 8, 4, 2][yolo_index]
            layers = mdef['from'] if 'from' in mdef else []
            modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']],
                                nc=mdef['classes'],
                                img_size=img_size,
                                yolo_index=yolo_index,
                                layers=layers,
                                stride=stride)

            try:
                bo = -4.5
                bc = math.log(1 / (modules.nc - 0.99))

                j = layers[yolo_index] if 'from' in mdef else -1
                bias_ = module_list[j][0].bias
                bias = bias_[:modules.no * modules.na].view(modules.na, -1)
                bias[:, 4] += bo - bias[:, 4].mean()
                bias[:, 5:] += bc - bias[:, 5:].mean()
                module_list[j][0].bias = torch.nn.Parameter(
                    bias_, requires_grad=bias_.requires_grad)
            except:
                print('WARNING: smart bias initialization failure.')

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index
        self.layers = layers
        self.stride = stride
        self.nl = len(layers)
        self.na = len(anchors)
        self.nc = nc
        self.no = nc + 5
        self.nx, self.ny = 0, 0
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng
        self.ng = torch.Tensor(ng).to(device)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid(
                [torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view(
                (1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p, img_size, out):
        ASFF = False
        if ASFF:
            i, n = self.index, self.nl
            p = out[self.layers[i]]
            bs, _, ny, nx = p.shape
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

            w = torch.sigmoid(p[:, -n:]) * (2 / n)

            p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    p += w[:, j:j + 1] * \
                        F.interpolate(
                            out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)

        elif ONNX_EXPORT:
            bs = 1
        else:
            bs, _, ny, nx = p.shape
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(
            0, 1, 3, 4, 2).contiguous()

        if self.training:
            return p

        elif ONNX_EXPORT:

            m = self.na * self.nx * self.ny
            ng = 1 / self.ng.repeat((m, 1))
            grid = self.grid.repeat((1, self.na, 1, 1, 1)).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(
                (1, 1, self.nx, self.ny, 1)).view(m, 2) * ng

            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid
            wh = torch.exp(p[:, 2:4]) * anchor_wh
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
                torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])
            return p_cls, xy * ng, wh

        else:
            io = p.clone()
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p


class Darknet(nn.Module):

    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(
            self.module_defs, img_size)
        self.yolo_layers = get_yolo_layers(self)

        self.version = np.array([0, 2, 5], dtype=np.int32)
        self.seen = np.array([0], dtype=np.int64)
        self.info(verbose)

    def forward(self, x, augment=False, verbose=False):

        if not augment:
            return self.forward_once(x)
        else:
            img_size = x.shape[-2:]
            s = [0.83, 0.67]
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(
                                        x.flip(3), s[0], same_shape=False),
                                    torch_utils.scale_img(
                                        x, s[1], same_shape=False),
                                    )):

                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]
            y[1][..., 0] = img_size[1] - y[1][..., 0]
            y[2][..., :4] /= s[1]

            y = torch.cat(y, 1)
            return y, None

    def forward_once(self, x, augment=False, verbose=False):
        img_size = x.shape[-2:]
        yolo_out, out = [], []
        if verbose:
            print('0', x.shape)
            str = ''

        if augment:
            nb = x.shape[0]
            s = [0.83, 0.67]
            x = torch.cat((x,
                           torch_utils.scale_img(x.flip(3), s[0]),
                           torch_utils.scale_img(x, s[1]),
                           ), 0)

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ['WeightedFeatureFusion', 'FeatureConcat']:
                if verbose:
                    l = [i - 1] + module.layers
                    sh = [list(x.shape)] + [list(out[i].shape)
                                            for i in module.layers]
                    str = ' >> ' + \
                        ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)
            elif name == 'YOLOLayer':
                yolo_out.append(module(x, img_size, out))
            else:
                x = module(x)

            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' %
                      (i, len(self.module_list), name), list(x.shape), str)
                str = ''

        if self.training:
            return yolo_out
        elif ONNX_EXPORT:
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            return x[0], torch.cat(x[1:3], 1)
        else:
            x, p = zip(*yolo_out)
            x = torch.cat(x, 1)
            if augment:
                x = torch.split(x, nb, dim=0)
                x[1][..., :4] /= s[0]
                x[1][..., 0] = img_size[1] - x[1][..., 0]
                x[2][..., :4] /= s[1]
                x = torch.cat(x, 1)
            return x, p

    def fuse(self):

        print('Fusing layers...')
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        self.info()

    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)


def get_yolo_layers(model):
    return [i for i, x in enumerate(model.module_defs) if x['type'] == 'yolo']


def load_darknet_weights(self, weights, cutoff=-1):

    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    with open(weights, 'rb') as f:

        self.version = np.fromfile(f, dtype=np.int32, count=3)
        self.seen = np.fromfile(f, dtype=np.int64, count=1)

        weights = np.fromfile(f, dtype=np.float32)

    ptr = 0
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv = module[0]
            if mdef['batch_normalize']:

                bn = module[1]
                nb = bn.bias.numel()

                bn.bias.data.copy_(torch.from_numpy(
                    weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb

                bn.weight.data.copy_(torch.from_numpy(
                    weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb

                bn.running_mean.data.copy_(torch.from_numpy(
                    weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb

                bn.running_var.data.copy_(torch.from_numpy(
                    weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:

                nb = conv.bias.numel()
                conv_b = torch.from_numpy(
                    weights[ptr:ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb

            nw = conv.weight.numel()
            conv.weight.data.copy_(torch.from_numpy(
                weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw


def save_weights(self, path='model.weights', cutoff=-1):

    with open(path, 'wb') as f:

        self.version.tofile(f)
        self.seen.tofile(f)

        for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional':
                conv_layer = module[0]

                if mdef['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)

                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)

                conv_layer.weight.data.cpu().numpy().tofile(f)


def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights'):

    model = Darknet(cfg)

    if weights.endswith('.pt'):
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        save_weights(model, path='converted.weights', cutoff=-1)
        print("Success: converted '%s' to 'converted.weights'" % weights)

    elif weights.endswith('.weights'):
        _ = load_darknet_weights(model, weights)

        chkpt = {'epoch': -1,
                 'best_fitness': None,
                 'training_results': None,
                 'model': model.state_dict(),
                 'optimizer': None}

        torch.save(chkpt, 'converted.pt')
        print("Success: converted '%s' to 'converted.pt'" % weights)

    else:
        print('Error: extension not supported.')


def attempt_download(weights):

    msg = weights + ' missing, try downloading from https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0'

    if weights and not os.path.isfile(weights):
        d = {'yolov3-spp.weights': '16lYS4bcIdM2HdmyJBVDOvt3Trx6N3W2R',
             'yolov3.weights': '1uTlyDWlnaqXcsKOktP5aH_zRDbfcDp-y',
             'yolov3-tiny.weights': '1CCF-iNIIkYesIDzaPvdwlcf7H9zSsKZQ',
             'yolov3-spp.pt': '1f6Ovy3BSq2wYq4UfvFUpxJFNDFfrIDcR',
             'yolov3.pt': '1SHNFyoe5Ni8DajDNEqgB2oVKBb_NoEad',
             'yolov3-tiny.pt': '10m_3MlpQwRtZetQxtksm9jqHrPTHZ6vo',
             'darknet53.conv.74': '1WUVBid-XuoUBmvzBVUCBl_ELrzqwA8dJ',
             'yolov3-tiny.conv.15': '1Bw0kCpplxUqyRYAJr9RY9SGnOJbo9nEj',
             'yolov3-spp-ultralytics.pt': '1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4'}

        file = Path(weights).name
        if file in d:
            r = gdrive_download(id=d[file], name=weights)
        else:
            url = 'https://pjreddie.com/media/files/' + file
            print('Downloading ' + url)
            r = os.system('curl -f ' + url + ' -o ' + weights)

        if not (r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6):
            os.system('rm ' + weights)
            raise Exception(msg)
