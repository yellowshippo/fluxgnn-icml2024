
import dataclasses as dc

import siml


@dc.dataclass
class SimulationSetting(siml.setting.TypedDataClass):
    start_t: float = 0.
    t_max: float = 0.
    delta_t: float = 0.
    original_delta_t: float = 0.

    shift_t: float = None
    max_timeseries_t: float = None  # Relative to start_t
    eval_t_max: float = None  # Relative to start_t
    test_t_max: float = None  # Relative to start_t

    def __post_init__(self):
        if self.max_timeseries_t is None:
            self.max_timeseries_t = self.t_max

        if self.shift_t is None:
            self.shift_t = self.max_timeseries_t

        if self.eval_t_max is None:
            self.eval_t_max = self.t_max

        if self.test_t_max is None:
            self.test_t_max = self.eval_t_max

        return


@dc.dataclass
class MLSetting(siml.setting.TypedDataClass):
    seed: int = 0

    n_epoch: int = 10
    lr: float = 1.e-3
    n_continue: int = 5
    dict_loss_function_name: dict[str, str] = None
    factor_loss_upper_limit: float = 100.
    dict_loss_weight: dict[str, float] = dc.field(default_factory=lambda: {})
    clip: float = dc.field(
        default=None, metadata={'allow_none': True})
    update_period: int = 1
    stop_trigger_epoch: int = 10
    patience: int = 3

    show_weights: bool = False
    show_grads: bool = False

    def __post_init__(self):
        if self.dict_loss_function_name is None:
            self.dict_loss_function_name = {
                k: 'relative_l2' for k in self.dict_loss_weight.keys()}
        return


@dc.dataclass
class ModelSetting(siml.setting.TypedDataClass):
    mode: str
    nodes: list[int] = dc.field(default_factory=lambda: [16, 16])
    bias: bool = False
    interpolation_bias: bool = False
    activations: list[str] = dc.field(default_factory=lambda: ['identity'])

    use_mlp: bool = False
    diff: bool = True
    normalize: bool = False
    n_forward: int = 1
    deep_processor: bool = False
    n_repeat_deep_processor: int = 0

    trainable_bundle: bool = False
    n_bundle: int = 1
    tb_base: float = 0.
    scale_encdec: bool = False
    l2_for_scale_encdec: bool = False
    shared_temporal_encdec: bool = False
    time_average: bool = False
    time_average_depth: int = 0
    time_average_coeff: float = .7

    sqrt: bool = True
    center_data: bool = False
    encoded_pushforward: bool = False
    coeff_processor: float = 1.
    property_for_ml: float = False

    upwind_only: bool = False
    trainable_u_interpolation: bool = True
    trainable_alpha_interpolation: bool = True
    trainable_u_interpolation_for_alpha: bool = False
    shared_u_interpolation_for_alpha: bool = False

    trainable_u_convection: bool = True
    trainable_alpha_convection: bool = True
    trainable_u_diffusion: bool = True
    trainable_alpha_diffusion: bool = True

    positive_encoder: bool = True
    encoder_weight_filter: str = 'identity'
    encoded_pressure: bool = True
    positive: bool = False
    positive_weight_method: str = 'sigmoid'  # or 'square', 'shifted_tanh'
    train_rho_mlp: bool = False
    train_grad_rho_mlp: bool = False
    ml_interpolation: str = 'separate'
    # or 'concatenate', 'upwind', 'upwind_center', 'upwind_center_concatenate',
    #    'bounded', 'non_ml'

    fv_residual: bool = False
    fv_residual_nonlinear_weight: float = 0.5

    residual_processor: bool = False
    residual_processor_nonlinear_weight: float = 0.5

    show_scale: bool = False
