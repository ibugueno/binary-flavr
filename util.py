import torch.nn as nn
import numpy

class BinOp():
    def __init__(self, model):
        # count the number of Conv2d and Linear
        count_targets = 0
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                count_targets = count_targets + 1

        start_range = 1
        end_range = count_targets-2
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_modules = []
        index = -1
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)
        return

    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                    mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                    self.target_modules[index].data.clamp(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
        for index in range(self.num_of_params):
            w = self.target_modules[index].data
            s = w.size()
            n = w[0].nelement()

            if len(s) == 4:  # Conv2d
                m = w.norm(1, 3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n)

            elif len(s) == 5:  # Conv3d
                m = w.norm(1, 4, keepdim=True)\
                    .sum(3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n)

            elif len(s) == 2:  # Linear
                m = w.norm(1, 1, keepdim=True).div(n)

            else:
                raise ValueError(f"Unsupported tensor shape: {s}")

            self.target_modules[index].data = w.sign().mul(m.expand_as(w))


    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            grad = self.target_modules[index].grad.data
            s = weight.size()
            n = weight[0].nelement()

            # Step 1: calcula m con gradiente en regiones no saturadas
            if len(s) == 4:  # Conv2d
                m = weight.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 5:  # Conv3d
                m = weight.norm(1, 4, keepdim=True)\
                        .sum(3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:  # Linear
                m = weight.norm(1, 1, keepdim=True).div(n).expand(s)
            else:
                raise ValueError(f"Unsupported tensor shape: {s}")

            m[weight.lt(-1.0)] = 0
            m[weight.gt(1.0)] = 0
            m = m.mul(grad)

            # Step 2: compute m_add
            m_add = weight.sign().mul(grad)

            if len(s) == 4:  # Conv2d
                m_add = m_add.sum(3, keepdim=True)\
                            .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 5:  # Conv3d
                m_add = m_add.sum(4, keepdim=True)\
                            .sum(3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:  # Linear
                m_add = m_add.sum(1, keepdim=True).div(n).expand(s)

            m_add = m_add.mul(weight.sign())

            # Step 3: final grad update
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0 - 1.0 / s[1]).mul(n)

