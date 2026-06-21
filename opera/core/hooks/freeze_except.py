from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


def _unwrap_model(model):
    return model.module if is_module_wrapper(model) else model


def _get_submodule(module, name):
    current = module
    for part in name.split('.'):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


@HOOKS.register_module()
class FreezeExceptHook(Hook):
    """Freeze all parameters and module modes except selected prefixes."""

    def __init__(self,
                 trainable_param_prefixes,
                 trainable_module_prefixes=None,
                 set_frozen_eval=True):
        self.trainable_param_prefixes = tuple(trainable_param_prefixes)
        self.trainable_module_prefixes = tuple(
            trainable_module_prefixes or trainable_param_prefixes)
        self.set_frozen_eval = set_frozen_eval
        self._logged = False

    def before_run(self, runner):
        model = _unwrap_model(runner.model)
        trainable_numel = 0
        frozen_numel = 0
        trainable_names = []
        for name, param in model.named_parameters():
            is_trainable = name.startswith(self.trainable_param_prefixes)
            param.requires_grad_(is_trainable)
            if is_trainable:
                trainable_numel += param.numel()
                trainable_names.append(name)
            else:
                frozen_numel += param.numel()

        if runner.logger is not None:
            runner.logger.info(
                'FreezeExceptHook: trainable params=%d tensors/%d scalars; '
                'frozen scalars=%d; prefixes=%s',
                len(trainable_names), trainable_numel, frozen_numel,
                self.trainable_param_prefixes)

    def _is_trainable_related_module(self, name):
        if name == '':
            return True
        for prefix in self.trainable_module_prefixes:
            if name == prefix:
                return True
            if name.startswith(prefix + '.'):
                return True
            if prefix.startswith(name + '.'):
                return True
        return False

    def before_train_epoch(self, runner):
        if not self.set_frozen_eval:
            return

        model = _unwrap_model(runner.model)
        for name, module in model.named_modules():
            if not self._is_trainable_related_module(name):
                module.eval()
        for prefix in self.trainable_module_prefixes:
            _get_submodule(model, prefix).train()

        if not self._logged and runner.logger is not None:
            runner.logger.info(
                'FreezeExceptHook: frozen modules set to eval; train modules=%s',
                self.trainable_module_prefixes)
            self._logged = True
