
import copy

import einops
import numpy as np
import siml
import torch

from .util import torch_tensor
from .util import cat_tail


# class DGGNN(siml.networks.abstract_equivariant_gnn.AbstractEquivariantGNN):
class FluxGNN(siml.networks.siml_module.SimlModule):
    """DGGNN block."""

    @staticmethod
    def get_name():
        return 'dggnn'

    @staticmethod
    def accepts_multiple_inputs():
        return True

    @staticmethod
    def is_trainable():
        return True

    @staticmethod
    def uses_support():
        return False

    @classmethod
    def _get_n_input_node(
            cls, block_setting, predecessors, dict_block_setting,
            input_length, **kwargs):
        return np.max([
            dict_block_setting[predecessor].nodes[-1]
            for predecessor in predecessors])

    def __init__(self, block_setting, passive_names=None):
        block_setting.optional['create_subchain'] = False
        block_setting.optional['set_last_activation'] = False
        block_setting.optional['skip_propagation_function'] = True

        super().__init__(
            block_setting, create_linears=False, no_parameter=False,
            create_dropouts=False)
        self.mode = self.block_setting.optional['propagations']
        self.use_mlp = self.block_setting.optional.get('use_mlp', False)
        self.trainable = self.block_setting.optional.get('trainable', True)

        self.bias = self.block_setting.bias
        self.interpolation_bias = self.block_setting.optional.get(
            'interpolation_bias', False)
        print(f"Interpolation bias: {self.interpolation_bias}")
        self.coeff_processor = self.block_setting.optional.get(
            'coeff_processor', 1.)
        if abs(self.coeff_processor - 1.) > 1e-3:
            print(f"Coeff for processor: {self.coeff_processor}")

        self.encoded_pressure = self.block_setting.optional.get(
            'encoded_pressure', True)
        self.positive_encoder = self.block_setting.optional.get(
            'positive_encoder', True)
        self.encoder_weight_filter = self.block_setting.optional.get(
            'encoder_weight_filter', 'identity')
        self.positive = self.block_setting.optional.get('positive', False)

        self.diff = self.block_setting.optional.get('diff', False)
        self.train_rho_mlp = self.block_setting.optional.get(
            'train_rho_mlp', False)
        self.train_grad_rho_mlp = self.block_setting.optional.get(
            'train_grad_rho_mlp', False)
        self.ml_interpolation = self.block_setting.optional.get(
            'ml_interpolation', 'separate')

        self.deep_processor = self.block_setting.optional.get(
            'deep_processor', False)
        self.n_repeat_deep_processor = self.block_setting.optional.get(
            'n_repeat_deep_processor', 0)
        if self.deep_processor:
            if self.n_repeat_deep_processor < 1:
                raise ValueError(
                    'Set n_repeat_deep_processor > 0 '
                    'when deep_processor = True')
            print('Deep processor applied with the following setting:')
            print(f"- n_repeat_deep_processor: {self.n_repeat_deep_processor}")

        self.n_forward = self.block_setting.optional.get(
            'n_forward', 1)
        if self.n_forward < 1:
            raise ValueError(f"Invalid n_forward: {self.n_forward}")
        self.n_bundle = self.block_setting.optional.get(
            'n_bundle', 1)
        self.trainable_bundle = self.block_setting.optional.get(
            'trainable_bundle', False)
        self.tb_base = self.block_setting.optional.get(
            'tb_base', 0.)
        self.shared_temporal_encdec = self.block_setting.optional.get(
            'shared_temporal_encdec', False)
        self.scale_encdec = self.block_setting.optional.get(
            'scale_encdec', False)
        self.l2_for_scale_encdec = self.block_setting.optional.get(
            'l2_for_scale_encdec', False)
        if self.block_setting.nodes[0] % self.n_bundle != 0:
            raise ValueError(
                '# of nodes should be multiple of n_bundle '
                f"({self.block_setting.nodes[0] = } vs {self.n_bundle = }")
        if self.n_bundle > 1:
            self.temporal_bundling = True
            print('Temporal bundling with the following setting:')
            print(f"- n_bundle: {self.n_bundle}")
            print(f"- shared_temporal_encdec: {self.shared_temporal_encdec}")
            print(f"- scale_encdec: {self.scale_encdec}")
            print(f"- l2_for_scale_encdec: {self.l2_for_scale_encdec}")
        else:
            self.temporal_bundling = False
        self.time_average = self.block_setting.optional.get(
            'time_average', False)
        self.time_average_depth = self.block_setting.optional.get(
            'time_average_depth', 0)
        if self.trainable_bundle:
            self.time_average_coeff = self.block_setting.optional.get(
                'time_average_coeff', None)
        else:
            self.time_average_coeff = self.block_setting.optional.get(
                'time_average_coeff', .7)
        if self.time_average:
            if self.time_average_depth < 1:
                raise ValueError(
                    'Set time_average_depth > 0 when time_average = True')
            print('Time average with the following setting:')
            print(f"- time_average_depth: {self.time_average_depth}")
            if self.trainable_bundle:
                print('- time_average_coeff: trainable')
                print(f"- tb_base: {self.tb_base}")
            else:
                print(f"- time_average_coeff: {self.time_average_coeff}")

        self.center_data = self.block_setting.optional.get(
            'center_data', False)
        self.encoded_pushforward = self.block_setting.optional.get(
            'encoded_pushforward', False)

        self.trainable_u_interpolation \
            = self.block_setting.optional.get(
                'trainable_u_interpolation', True)
        self.trainable_alpha_interpolation \
            = self.block_setting.optional.get(
                'trainable_alpha_interpolation', True)
        self.trainable_u_interpolation_for_alpha \
            = self.block_setting.optional.get(
                'trainable_u_interpolation_for_alpha', False)
        self.shared_u_interpolation_for_alpha \
            = self.block_setting.optional.get(
                'shared_u_interpolation_for_alpha', False)
        if self.trainable_u_interpolation_for_alpha \
                and self.shared_u_interpolation_for_alpha:
            raise ValueError(
                'trainable_u_interpolation_for_alpha and '
                'shared_u_interpolation_for_alpha cannot be True '
                'at the same time')

        self.trainable_u_convection = self.block_setting.optional.get(
            'trainable_u_convection', True)
        self.trainable_u_diffusion = self.block_setting.optional.get(
            'trainable_u_diffusion', True)
        self.trainable_alpha_convection = self.block_setting.optional.get(
            'trainable_alpha_convection', True)
        self.trainable_alpha_diffusion = self.block_setting.optional.get(
            'trainable_alpha_diffusion', True)

        self.fv_residual = self.block_setting.optional.get(
            'fv_residual', False)
        self.fv_residual_nonlinear_weight = self.block_setting.optional.get(
            'fv_residual_nonlinear_weight', .5)
        if self.fv_residual:
            print(
                f"FV Residual with coeff: {self.fv_residual_nonlinear_weight}")

        self.residual_processor = self.block_setting.optional.get(
            'residual_processor', False)
        self.residual_processor_nonlinear_weight = \
            self.block_setting.optional.get(
                'residual_processor_nonlinear_weight', .5)

        if passive_names is None:
            self.passive_names = []
        else:
            self.passive_names = passive_names

        self.dict_mlp = self._create_dict_mlp(block_setting)

        return

    def _create_dict_mlp(self, block_setting):

        if self.mode == 'mixture':
            return self._create_dict_mlp_mixture(block_setting)
        elif self.mode == 'convection_diffusion':
            return self._create_dict_mlp_convection_diffusion(
                block_setting)
        else:
            raise ValueError(f"Unexpected mode: {self.mode}")

    def _create_dict_mlp_mixture(self, block_setting):
        # Generate settings
        # # Encoder
        encode_u_setting = copy.deepcopy(block_setting)
        encode_u_setting.name = 'ENCODE_U'
        encode_u_setting.coeff = 1.
        encode_u_setting.no_grad = True
        encode_u_setting.bias = False
        encode_u_setting.residual = False
        encode_u_setting.nodes = [
            1, round(block_setting.nodes[-1] / self.n_bundle)]
        encode_u_setting.activations = ['identity']
        encode_u_setting.optional['positive_weight'] \
            = self.positive_encoder
        encode_u_setting.optional['weight_filter'] = self.encoder_weight_filter

        dirichlet_u_setting = copy.deepcopy(encode_u_setting)
        dirichlet_u_setting.name = 'DIRICHLET_U'
        dirichlet_u_setting.activations = ['identity']
        dirichlet_u_setting.no_grad = True

        neumann_u_setting = copy.deepcopy(encode_u_setting)
        neumann_u_setting.name = 'NEUMANN_U'
        neumann_u_setting.activations = ['identity']
        neumann_u_setting.no_grad = True

        # Pressure: M / (L T^2)
        encode_p_setting = copy.deepcopy(block_setting)
        encode_p_setting.name = 'ENCODE_P'
        encode_p_setting.coeff = 1.
        encode_p_setting.bias = False
        encode_p_setting.residual = False
        encode_p_setting.nodes = [
            1, round(block_setting.nodes[-1] / self.n_bundle)]
        encode_p_setting.optional['positive_weight'] \
            = self.positive_encoder
        encode_p_setting.optional['weight_filter'] = self.encoder_weight_filter
        encode_p_setting.activations = ['identity']

        dirichlet_p_setting = copy.deepcopy(encode_p_setting)
        dirichlet_p_setting.name = 'DIRICHLET_P'
        dirichlet_p_setting.activations = ['identity']
        dirichlet_p_setting.no_grad = True

        neumann_p_setting = copy.deepcopy(encode_p_setting)
        neumann_p_setting.name = 'NEUMANN_P'
        neumann_p_setting.activations = ['identity']
        neumann_p_setting.no_grad = True

        # Volume fraction: [1]
        encode_alpha_setting = copy.deepcopy(block_setting)
        encode_alpha_setting.name = 'ENCODE_ALPHA'
        encode_alpha_setting.coeff = 1.
        encode_alpha_setting.no_grad = True
        encode_alpha_setting.bias = False
        encode_alpha_setting.residual = False
        encode_alpha_setting.nodes = [
            1, round(block_setting.nodes[-1] / self.n_bundle)]
        encode_alpha_setting.activations = ['identity']
        encode_alpha_setting.optional['positive_weight'] \
            = self.positive_encoder
        encode_alpha_setting.optional['weight_filter'] \
            = self.encoder_weight_filter
        encode_alpha_setting.optional['dimension'] = {
            'length': 0, 'time': 0, 'mass': 0}

        dirichlet_alpha_setting = copy.deepcopy(encode_alpha_setting)
        dirichlet_alpha_setting.name = 'DIRICHLET_ALPHA'
        dirichlet_alpha_setting.activations = ['identity']
        dirichlet_alpha_setting.no_grad = True

        neumann_alpha_setting = copy.deepcopy(encode_alpha_setting)
        neumann_alpha_setting.name = 'NEUMANN_ALPHA'
        neumann_alpha_setting.activations = ['identity']
        neumann_alpha_setting.no_grad = True

        # Viscosity: L^2 T^{-1}
        encode_nu_setting = copy.deepcopy(block_setting)
        encode_nu_setting.name = 'ENCODE_NU'
        encode_nu_setting.coeff = 1.
        encode_nu_setting.bias = False
        encode_nu_setting.residual = False
        encode_nu_setting.nodes = [1, block_setting.nodes[-1]]
        encode_nu_setting.activations = ['identity']
        encode_nu_setting.optional['positive_weight'] = self.positive_encoder
        encode_nu_setting.optional['weight_filter'] \
            = self.encoder_weight_filter

        encode_rho_setting = copy.deepcopy(block_setting)
        encode_rho_setting.name = 'ENCODE_RHO'
        encode_rho_setting.coeff = 1.
        encode_rho_setting.bias = False
        encode_rho_setting.residual = False
        encode_rho_setting.nodes = [1, block_setting.nodes[-1]]
        encode_rho_setting.activations = ['identity']
        encode_rho_setting.optional['positive_weight'] = self.positive_encoder
        encode_rho_setting.optional['weight_filter'] \
            = self.encoder_weight_filter

        encode_gravity_setting = copy.deepcopy(block_setting)
        encode_gravity_setting.name = 'ENCODE_GRAVITY'
        encode_gravity_setting.coeff = 1.
        encode_gravity_setting.bias = False
        encode_gravity_setting.residual = False
        encode_gravity_setting.nodes = [1, block_setting.nodes[-1]]
        encode_gravity_setting.activations = ['identity']
        encode_gravity_setting.optional['positive_weight'] \
            = self.positive_encoder
        encode_gravity_setting.optional['weight_filter'] \
            = self.encoder_weight_filter

        encode_diffusion_alpha_setting = copy.deepcopy(block_setting)
        encode_diffusion_alpha_setting.name = 'ENCODE_DIFFUSION_ALPHA'
        encode_diffusion_alpha_setting.coeff = 1.
        encode_diffusion_alpha_setting.bias = False
        encode_diffusion_alpha_setting.residual = False
        encode_diffusion_alpha_setting.nodes = [1, block_setting.nodes[-1]]
        encode_diffusion_alpha_setting.activations = ['identity']
        encode_diffusion_alpha_setting.optional['positive_weight'] \
            = self.positive_encoder
        encode_diffusion_alpha_setting.optional['weight_filter'] \
            = self.encoder_weight_filter

        # # Processor

        # For rho u x u
        conv_u_setting = copy.deepcopy(block_setting)
        conv_u_setting.name = 'CONV_U'
        conv_u_setting.coeff = self.coeff_processor
        conv_u_setting.bias = self.bias
        conv_u_setting.residual = self.residual_processor
        conv_u_setting.optional['residual_weight'] = \
            self.residual_processor_nonlinear_weight
        conv_u_setting.optional['diff'] = self.diff
        conv_u_setting.optional['positive'] = self.positive
        # [rho] [u] [u] = (M / L^3) (L / T) (L / T) = M / (L T^2)
        conv_u_setting.optional['dimension'] = {
            'length': -1, 'time': -2, 'mass': 1}

        # For u (interpolation)
        int_u_setting = copy.deepcopy(block_setting)
        int_u_setting.name = 'INT_U'
        int_u_setting.coeff = self.coeff_processor
        int_u_setting.bias = self.interpolation_bias
        int_u_setting.residual = self.residual_processor
        int_u_setting.optional['residual_weight'] = \
            self.residual_processor_nonlinear_weight
        int_u_setting.optional['diff'] = self.diff
        int_u_setting.optional['positive'] = self.positive
        int_u_setting.optional['dimension'] = {
            'length': 1, 'time': -1, 'mass': 0}
        if self.ml_interpolation == 'concatenate':
            int_u_setting.nodes[0] = int_u_setting.nodes[0] * 3
        elif self.ml_interpolation == 'upwind_center_concatenate':
            int_u_setting.nodes[0] = int_u_setting.nodes[0] * 2
        elif self.ml_interpolation == 'bounded':
            int_u_setting.nodes[0] = int_u_setting.nodes[0] * 2
            int_u_setting.activations[-1] = 'sigmoid'
            int_u_setting.optional['invariant'] = True
        else:
            pass

        # For grad p (may not be used)
        # [p] = M / (L T^2)
        grad_p_setting = copy.deepcopy(block_setting)
        grad_p_setting.name = 'GRAD_P'
        grad_p_setting.coeff = self.coeff_processor
        grad_p_setting.bias = self.bias
        grad_p_setting.residual = self.residual_processor
        grad_p_setting.optional['residual_weight'] = \
            self.residual_processor_nonlinear_weight
        grad_p_setting.optional['diff'] = False
        grad_p_setting.optional['positive'] = self.positive
        grad_p_setting.optional['dimension'] = {
            'length': -2, 'time': -2, 'mass': 1}

        # For grad u_i
        diffusion_grad_setting = copy.deepcopy(block_setting)
        diffusion_grad_setting.name = 'DIFFUSION_GRAD_U'
        diffusion_grad_setting.coeff = self.coeff_processor
        diffusion_grad_setting.bias = self.bias
        diffusion_grad_setting.residual = self.residual_processor
        diffusion_grad_setting.optional['residual_weight'] = \
            self.residual_processor_nonlinear_weight
        diffusion_grad_setting.optional['diff'] = False
        diffusion_grad_setting.optional['positive'] = self.positive
        diffusion_grad_setting.optional['dimension'] = {
            'length': 0, 'time': -1, 'mass': 0}

        # # For alpha (volume fraction)
        # For alpha (interpolation)
        conv_alpha_setting = copy.deepcopy(block_setting)
        conv_alpha_setting.name = 'CONV_ALPHA'
        conv_alpha_setting.coeff = self.coeff_processor
        conv_alpha_setting.bias = self.interpolation_bias
        conv_alpha_setting.residual = self.residual_processor
        conv_alpha_setting.optional['residual_weight'] = \
            self.residual_processor_nonlinear_weight
        conv_alpha_setting.optional['diff'] = self.diff
        conv_alpha_setting.optional['positive'] = self.positive
        conv_alpha_setting.optional['dimension'] = {
            'length': 0, 'time': 0, 'mass': 0}
        if self.ml_interpolation == 'concatenate':
            conv_alpha_setting.nodes[0] = conv_alpha_setting.nodes[0] * 3
        elif self.ml_interpolation == 'upwind_center_concatenate':
            conv_alpha_setting.nodes[0] = conv_alpha_setting.nodes[0] * 2
        elif self.ml_interpolation == 'bounded':
            conv_alpha_setting.nodes[0] = conv_alpha_setting.nodes[0] * 2
            conv_alpha_setting.activations[-1] = 'sigmoid'
            conv_alpha_setting.optional['invariant'] = True
        else:
            pass

        # For u alpha (convection)
        conv_alpha_flux_setting = copy.deepcopy(block_setting)
        conv_alpha_flux_setting.name = 'CONV_ALPHA_FLUX'
        conv_alpha_flux_setting.coeff = self.coeff_processor
        conv_alpha_flux_setting.bias = self.bias
        conv_alpha_flux_setting.residual = self.residual_processor
        conv_alpha_flux_setting.optional['residual_weight'] = \
            self.residual_processor_nonlinear_weight
        conv_alpha_flux_setting.optional['diff'] = self.diff
        conv_alpha_flux_setting.optional['positive'] = self.positive
        conv_alpha_flux_setting.optional['dimension'] = {
            'length': 1, 'time': -1, 'mass': 0}

        # For grad alpha
        alpha_diffusion_grad_setting = copy.deepcopy(block_setting)
        alpha_diffusion_grad_setting.name = 'ALPHA_DIFFUSION_GRAD'
        alpha_diffusion_grad_setting.coeff = self.coeff_processor
        alpha_diffusion_grad_setting.bias = self.bias
        alpha_diffusion_grad_setting.residual = self.residual_processor
        alpha_diffusion_grad_setting.optional['residual_weight'] = \
            self.residual_processor_nonlinear_weight
        alpha_diffusion_grad_setting.optional['diff'] = False
        alpha_diffusion_grad_setting.optional['positive'] = self.positive
        alpha_diffusion_grad_setting.optional['dimension'] = {
            'length': 0, 'time': -1, 'mass': 0}

        # For grad rho (used for pressure equation source)
        rho_diffusion_grad_setting = copy.deepcopy(block_setting)
        rho_diffusion_grad_setting.residual = self.residual_processor
        rho_diffusion_grad_setting.coeff = self.coeff_processor
        rho_diffusion_grad_setting.bias = self.bias
        rho_diffusion_grad_setting.optional['residual_weight'] = \
            self.residual_processor_nonlinear_weight
        rho_diffusion_grad_setting.name = 'RHO_DIFFUSION_GRAD'
        rho_diffusion_grad_setting.optional['diff'] = False
        rho_diffusion_grad_setting.optional['positive'] = self.positive
        rho_diffusion_grad_setting.optional['dimension'] = {
            'length': -3 - 1, 'time': 0, 'mass': 1}

        # For rho (used for buoyancy)
        rho_mlp_setting = copy.deepcopy(block_setting)
        rho_mlp_setting.name = 'RHO_MLP'
        rho_mlp_setting.coeff = self.coeff_processor
        rho_mlp_setting.bias = self.bias
        rho_mlp_setting.residual = self.residual_processor
        rho_mlp_setting.optional['residual_weight'] = \
            self.residual_processor_nonlinear_weight
        rho_mlp_setting.optional['diff'] = self.diff
        rho_mlp_setting.optional['positive'] = self.positive
        rho_mlp_setting.optional['dimension'] = {
            'length': -3, 'time': 0, 'mass': 1}

        # # Temporal bundling
        tb_u_setting = copy.deepcopy(block_setting)
        tb_u_setting.name = 'TB_U'
        tb_u_setting.optional['diff'] = False
        tb_u_setting.optional['positive'] = False
        tb_u_setting.nodes = [1, 1]
        tb_u_setting.activations = ['identity']

        tb_p_setting = copy.deepcopy(block_setting)
        tb_p_setting.name = 'TB_P'
        tb_p_setting.optional['diff'] = False
        tb_p_setting.optional['positive'] = False
        tb_p_setting.nodes = [1, 1]
        tb_p_setting.activations = ['identity']

        tb_alpha_setting = copy.deepcopy(block_setting)
        tb_alpha_setting.name = 'TB_ALPHA'
        tb_alpha_setting.optional['diff'] = False
        tb_alpha_setting.optional['positive'] = False
        tb_alpha_setting.nodes = [1, 1]
        tb_alpha_setting.activations = ['identity']

        # # Decoder
        inv_u_setting = copy.deepcopy(block_setting)
        inv_u_setting.name = 'DECODE_U'
        inv_u_setting.coeff = 1.
        inv_u_setting.no_grad = False
        inv_u_setting.residual = False

        inv_u_setting.activations = ['identity']

        inv_p_setting = copy.deepcopy(block_setting)
        inv_p_setting.name = 'DECODE_P'
        inv_p_setting.coeff = 1.
        inv_p_setting.no_grad = False
        inv_p_setting.residual = False
        inv_p_setting.activations = ['identity']

        inv_alpha_setting = copy.deepcopy(block_setting)
        inv_alpha_setting.name = 'DECODE_ALPHA'
        inv_alpha_setting.coeff = 1.
        inv_alpha_setting.no_grad = False
        inv_alpha_setting.residual = False
        inv_alpha_setting.activations = ['identity']

        dict_inv_passive_setting = {}
        for name in self.passive_names:
            inv_passive_setting = copy.deepcopy(block_setting)
            inv_passive_setting.name = f"DECODE_{name.upper()}"
            inv_passive_setting.coeff = 1.
            inv_passive_setting.no_grad = False
            inv_passive_setting.residual = False
            inv_passive_setting.activations = ['identity']
            dict_inv_passive_setting.update({name: inv_passive_setting})

        # Create models
        # # Encoder
        if self.temporal_bundling:
            if self.shared_temporal_encdec:
                _encode_u_mlp = siml.networks.proportional.Proportional(
                    encode_u_setting)
                encode_u_mlp = torch.nn.ModuleList([
                    _encode_u_mlp for _ in range(self.n_bundle)])
                dirichlet_u_mlp = torch.nn.ModuleList([
                    siml.networks.share.Share(
                        dirichlet_u_setting, reference_block=m)
                    for m in encode_u_mlp])
                neumann_u_mlp = torch.nn.ModuleList([
                    siml.networks.share.Share(
                        neumann_u_setting, reference_block=m)
                    for m in encode_u_mlp])

                _encode_p_mlp = siml.networks.proportional.Proportional(
                    encode_p_setting)
                encode_p_mlp = torch.nn.ModuleList([
                    _encode_p_mlp for _ in range(self.n_bundle)])
                dirichlet_p_mlp = torch.nn.ModuleList([
                    siml.networks.share.Share(
                        dirichlet_p_setting, reference_block=m)
                    for m in encode_p_mlp])
                neumann_p_mlp = torch.nn.ModuleList([
                    siml.networks.share.Share(
                        neumann_p_setting, reference_block=m)
                    for m in encode_p_mlp])

                _encode_alpha_mlp = siml.networks.proportional.Proportional(
                    encode_alpha_setting)
                encode_alpha_mlp = torch.nn.ModuleList([
                    _encode_alpha_mlp
                    for _ in range(self.n_bundle)])
                dirichlet_alpha_mlp = torch.nn.ModuleList([
                    siml.networks.share.Share(
                        dirichlet_alpha_setting, reference_block=m)
                    for m in encode_alpha_mlp])
                neumann_alpha_mlp = torch.nn.ModuleList([
                    siml.networks.share.Share(
                        neumann_alpha_setting, reference_block=m)
                    for m in encode_alpha_mlp])
            else:
                encode_u_mlp = torch.nn.ModuleList([
                    siml.networks.proportional.Proportional(encode_u_setting)
                    for _ in range(self.n_bundle)])
                dirichlet_u_mlp = torch.nn.ModuleList([
                    siml.networks.share.Share(
                        dirichlet_u_setting, reference_block=m)
                    for m in encode_u_mlp])
                neumann_u_mlp = torch.nn.ModuleList([
                    siml.networks.share.Share(
                        neumann_u_setting, reference_block=m)
                    for m in encode_u_mlp])

                encode_p_mlp = torch.nn.ModuleList([
                    siml.networks.proportional.Proportional(
                        encode_p_setting)
                    for _ in range(self.n_bundle)])
                dirichlet_p_mlp = torch.nn.ModuleList([
                    siml.networks.share.Share(
                        dirichlet_p_setting, reference_block=m)
                    for m in encode_p_mlp])
                neumann_p_mlp = torch.nn.ModuleList([
                    siml.networks.share.Share(
                        neumann_p_setting, reference_block=m)
                    for m in encode_p_mlp])

                encode_alpha_mlp = torch.nn.ModuleList([
                    siml.networks.proportional.Proportional(
                        encode_alpha_setting)
                    for _ in range(self.n_bundle)])
                dirichlet_alpha_mlp = torch.nn.ModuleList([
                    siml.networks.share.Share(
                        dirichlet_alpha_setting, reference_block=m)
                    for m in encode_alpha_mlp])
                neumann_alpha_mlp = torch.nn.ModuleList([
                    siml.networks.share.Share(
                        neumann_alpha_setting, reference_block=m)
                    for m in encode_alpha_mlp])
        else:
            encode_u_mlp = siml.networks.proportional.Proportional(
                encode_u_setting)
            dirichlet_u_mlp = siml.networks.share.Share(
                dirichlet_u_setting, reference_block=encode_u_mlp)
            neumann_u_mlp = siml.networks.share.Share(
                neumann_u_setting, reference_block=encode_u_mlp)

            encode_p_mlp = siml.networks.proportional.Proportional(
                encode_p_setting)
            dirichlet_p_mlp = siml.networks.share.Share(
                dirichlet_p_setting, reference_block=encode_p_mlp)
            neumann_p_mlp = siml.networks.share.Share(
                neumann_p_setting, reference_block=encode_p_mlp)

            encode_alpha_mlp = siml.networks.proportional.Proportional(
                encode_alpha_setting)
            dirichlet_alpha_mlp = siml.networks.share.Share(
                dirichlet_alpha_setting, reference_block=encode_alpha_mlp)
            neumann_alpha_mlp = siml.networks.share.Share(
                neumann_alpha_setting, reference_block=encode_alpha_mlp)

        encode_nu_mlp = siml.networks.proportional.Proportional(
            encode_nu_setting)
        encode_rho_mlp = siml.networks.proportional.Proportional(
            encode_rho_setting)
        encode_gravity_mlp = siml.networks.proportional.Proportional(
            encode_gravity_setting)
        encode_diffusion_alpha_mlp = siml.networks.proportional.Proportional(
            encode_diffusion_alpha_setting)

        # # Decoder
        if self.temporal_bundling:
            inv_u_mlp = torch.nn.ModuleList([
                siml.networks.pinv_mlp.PInvMLP(
                    inv_u_setting, reference_block=m)
                for m in encode_u_mlp])
            inv_p_mlp = torch.nn.ModuleList([
                siml.networks.pinv_mlp.PInvMLP(
                    inv_p_setting, reference_block=m)
                for m in encode_p_mlp])
            inv_alpha_mlp = torch.nn.ModuleList([
                siml.networks.pinv_mlp.PInvMLP(
                    inv_alpha_setting, reference_block=m)
                for m in encode_alpha_mlp])
        else:
            inv_u_mlp = siml.networks.pinv_mlp.PInvMLP(
                inv_u_setting, reference_block=encode_u_mlp)
            inv_p_mlp = siml.networks.pinv_mlp.PInvMLP(
                inv_p_setting, reference_block=encode_p_mlp)
            inv_alpha_mlp = siml.networks.pinv_mlp.PInvMLP(
                inv_alpha_setting, reference_block=encode_alpha_mlp)

        id_ = siml.networks.identity.Identity(
            siml.setting.BlockSetting())

        # Processor
        def _create_dict_mlp_mixture_processors(label=None):
            if self.use_mlp:
                conv_u_mlp = siml.networks.tensor_operations.EquivariantMLP(
                    conv_u_setting)

                if self.trainable_u_interpolation:
                    u_int_upwind = \
                        siml.networks.tensor_operations.EquivariantMLP(
                            int_u_setting)
                    u_int_both = \
                        siml.networks.tensor_operations.EquivariantMLP(
                            int_u_setting)
                    u_int_center = \
                        siml.networks.tensor_operations.EquivariantMLP(
                            int_u_setting)

                if self.trainable_u_interpolation_for_alpha:
                    u_alpha_int_upwind \
                        = siml.networks.tensor_operations.EquivariantMLP(
                            int_u_setting)
                    u_alpha_int_both \
                        = siml.networks.tensor_operations.EquivariantMLP(
                            int_u_setting)
                    u_alpha_int_center \
                        = siml.networks.tensor_operations.EquivariantMLP(
                            int_u_setting)
                elif self.shared_u_interpolation_for_alpha:
                    u_alpha_int_upwind = u_int_upwind
                    u_alpha_int_both = u_int_both
                    u_alpha_int_center = u_int_center
                else:
                    pass

                grad_p_mlp = siml.networks.tensor_operations.EquivariantMLP(
                    grad_p_setting)
                diffusion_grad_mlp \
                    = siml.networks.tensor_operations.EquivariantMLP(
                        diffusion_grad_setting)

                if self.trainable_alpha_interpolation:
                    alpha_int_upwind = siml.networks.mlp.MLP(
                        conv_alpha_setting)
                    alpha_int_both = siml.networks.mlp.MLP(
                        conv_alpha_setting)
                    alpha_int_center = siml.networks.mlp.MLP(
                        conv_alpha_setting)

                conv_alpha_flux_mlp \
                    = siml.networks.tensor_operations.EquivariantMLP(
                        conv_alpha_flux_setting)
                alpha_diffusion_grad_mlp \
                    = siml.networks.tensor_operations.EquivariantMLP(
                        alpha_diffusion_grad_setting)

                rho_diffusion_grad_mlp \
                    = siml.networks.tensor_operations.EquivariantMLP(
                        rho_diffusion_grad_setting)
                rho_mlp \
                    = siml.networks.tensor_operations.EquivariantMLP(
                        rho_mlp_setting)

            else:
                conv_u_mlp = siml.networks.tensor_operations.EnSEquivariantMLP(
                    conv_u_setting)

                if self.trainable_u_interpolation:
                    u_int_upwind \
                        = siml.networks.tensor_operations.EnSEquivariantMLP(
                            int_u_setting)
                    u_int_both \
                        = siml.networks.tensor_operations.EnSEquivariantMLP(
                            int_u_setting)
                    u_int_center \
                        = siml.networks.tensor_operations.EnSEquivariantMLP(
                            int_u_setting)

                if self.trainable_u_interpolation_for_alpha:
                    u_alpha_int_upwind \
                        = siml.networks.tensor_operations.EnSEquivariantMLP(
                            int_u_setting)
                    u_alpha_int_both \
                        = siml.networks.tensor_operations.EnSEquivariantMLP(
                            int_u_setting)
                    u_alpha_int_center \
                        = siml.networks.tensor_operations.EnSEquivariantMLP(
                            int_u_setting)
                elif self.shared_u_interpolation_for_alpha:
                    u_alpha_int_upwind = u_int_upwind
                    u_alpha_int_both = u_int_both
                    u_alpha_int_center = u_int_center
                else:
                    pass

                grad_p_mlp = siml.networks.tensor_operations.EnSEquivariantMLP(
                    grad_p_setting)
                diffusion_grad_mlp \
                    = siml.networks.tensor_operations.EnSEquivariantMLP(
                        diffusion_grad_setting)

                if self.trainable_alpha_interpolation:
                    alpha_int_upwind = siml.networks.mlp.MLP(
                        conv_alpha_setting)
                    alpha_int_both = siml.networks.mlp.MLP(
                        conv_alpha_setting)
                    alpha_int_center = siml.networks.mlp.MLP(
                        conv_alpha_setting)

                conv_alpha_flux_mlp \
                    = siml.networks.tensor_operations.EnSEquivariantMLP(
                        conv_alpha_flux_setting)
                alpha_diffusion_grad_mlp \
                    = siml.networks.tensor_operations.EnSEquivariantMLP(
                        alpha_diffusion_grad_setting)

                rho_diffusion_grad_mlp \
                    = siml.networks.tensor_operations.EnSEquivariantMLP(
                        rho_diffusion_grad_setting)
                rho_mlp \
                    = siml.networks.tensor_operations.EnSEquivariantMLP(
                        rho_mlp_setting)

            grad_p_mlp = id_
            if not self.train_grad_rho_mlp:
                rho_diffusion_grad_mlp = id_
            if not self.train_rho_mlp:
                rho_mlp = id_

            if self.ml_interpolation == 'upwind':
                u_int_both = None
                u_int_center = None
                u_alpha_int_both = None
                u_alpha_int_center = None
                alpha_int_both = None
                alpha_int_center = None

            elif 'concat' in self.ml_interpolation:
                u_int_center = None
                u_int_upwind = None
                u_alpha_int_center = None
                u_alpha_int_upwind = None
                alpha_int_center = None
                alpha_int_upwind = None

            elif self.ml_interpolation == 'bounded':
                u_int_center = None
                u_int_upwind = None
                u_alpha_int_center = None
                u_alpha_int_upwind = None
                alpha_int_center = None
                alpha_int_upwind = None

            elif self.ml_interpolation == 'upwind_center':
                u_int_both = None
                u_alpha_int_both = None
                alpha_int_both = None

            else:
                pass

            if not self.trainable_u_interpolation:
                u_int_upwind = None
                u_int_both = None
                u_int_center = None

            if not (
                    self.trainable_u_interpolation_for_alpha
                    or self.shared_u_interpolation_for_alpha):
                u_alpha_int_upwind = None
                u_alpha_int_both = None
                u_alpha_int_center = None

            if not self.trainable_alpha_interpolation:
                alpha_int_upwind = None
                alpha_int_both = None
                alpha_int_center = None

            if not self.trainable_u_convection:
                conv_u_mlp = id_

            if not self.trainable_u_diffusion:
                diffusion_grad_mlp = id_

            if not self.trainable_alpha_convection:
                conv_alpha_flux_mlp = id_

            if not self.trainable_alpha_diffusion:
                alpha_diffusion_grad_mlp = id_

            if label is None:
                label = ''

            return {
                f"{self.mode}{label}/conv_u_mlp": conv_u_mlp,
                f"{self.mode}{label}/diffusion_grad_mlp": diffusion_grad_mlp,
                f"{self.mode}{label}/grad_p_mlp": grad_p_mlp,

                f"{self.mode}{label}/interpolation_both": u_int_both,
                f"{self.mode}{label}/interpolation_upwind": u_int_upwind,
                f"{self.mode}{label}/interpolation_center": u_int_center,

                f"u_for_alpha{label}/interpolation_both": u_alpha_int_both,
                f"u_for_alpha{label}/interpolation_upwind": u_alpha_int_upwind,
                f"u_for_alpha{label}/interpolation_center": u_alpha_int_center,

                f"alpha{label}/conv_phi_flux_mlp": conv_alpha_flux_mlp,
                f"alpha{label}/diffusion_grad_mlp": alpha_diffusion_grad_mlp,
                f"alpha{label}/interpolation_both": alpha_int_both,
                f"alpha{label}/interpolation_upwind": alpha_int_upwind,
                f"alpha{label}/interpolation_center": alpha_int_center,

                f"rho{label}/diffusion_grad_mlp": rho_diffusion_grad_mlp,
                f"rho{label}/rho": rho_mlp,
            }

        # Temporal bundling
        def _create_dict_mlp_temporal_bundling(label):
            if not self.trainable_bundle:
                tb_u_mlp = id_
                tb_p_mlp = id_
                tb_alpha_mlp = id_
                return {
                    f"{self.mode}{label}/tb_u": tb_u_mlp,
                    f"{self.mode}{label}/tb_p": tb_p_mlp,
                    f"{self.mode}{label}/tb_alpha": tb_alpha_mlp,
                }

            else:
                tb_u_mlp = siml.networks.mlp.MLP(tb_u_setting)
                tb_p_mlp = siml.networks.mlp.MLP(tb_p_setting)
                tb_alpha_mlp = siml.networks.mlp.MLP(tb_alpha_setting)

                return {
                    f"{self.mode}{label}/tb_u": tb_u_mlp,
                    f"{self.mode}{label}/tb_p": tb_p_mlp,
                    f"{self.mode}{label}/tb_alpha": tb_alpha_mlp,
                }

        # Disable rho encoder to keep gradient valid
        encode_rho_mlp = id_

        if self.deep_processor:
            dict_processors = {}
            for i in range(self.n_repeat_deep_processor):
                dict_processors.update(
                    _create_dict_mlp_mixture_processors(label=i))
        else:
            dict_processors = _create_dict_mlp_mixture_processors()

        dict_tb = {}
        for i in range(self.time_average_depth):
            dict_tb.update(
                _create_dict_mlp_temporal_bundling(label=i))

        if not self.encoded_pressure:
            encode_p_mlp = id_
            dirichlet_p_mlp = id_
            neumann_p_mlp = id_
            encode_gravity_mlp = id_

        dict_mlp = {
            # Encoders
            f"{self.mode}/encode_u_mlp": encode_u_mlp,
            f"{self.mode}/dirichlet_u_mlp": dirichlet_u_mlp,
            f"{self.mode}/neumann_u_mlp": neumann_u_mlp,

            f"{self.mode}/encode_p_mlp": encode_p_mlp,
            f"{self.mode}/dirichlet_p_mlp": dirichlet_p_mlp,
            f"{self.mode}/neumann_p_mlp": neumann_p_mlp,

            f"{self.mode}/encode_alpha_mlp": encode_alpha_mlp,
            f"{self.mode}/dirichlet_alpha_mlp": dirichlet_alpha_mlp,
            f"{self.mode}/neumann_alpha_mlp": neumann_alpha_mlp,

            f"{self.mode}/encode_nu_mlp": encode_nu_mlp,
            f"{self.mode}/encode_rho_mlp": encode_rho_mlp,
            f"{self.mode}/encode_gravity_mlp": encode_gravity_mlp,
            f"{self.mode}/encode_diffusion_alpha_mlp":
            encode_diffusion_alpha_mlp,

            # Decoders
            f"{self.mode}/inv_u_mlp": inv_u_mlp,
            f"{self.mode}/inv_p_mlp": inv_p_mlp,
            f"{self.mode}/inv_alpha_mlp": inv_alpha_mlp,
        }
        dict_mlp.update(dict_processors)
        dict_mlp.update(dict_tb)

        moduledict_mlp = torch.nn.ModuleDict(dict_mlp)

        return moduledict_mlp

    def _create_dict_mlp_convection_diffusion(self, block_setting):

        # Encoder
        encode_u_setting = copy.deepcopy(block_setting)
        encode_u_setting.name = 'ENCODE_U'
        encode_u_setting.coeff = 1.
        encode_u_setting.nodes = [1] + block_setting.nodes[1:]
        encode_u_setting.optional['diff'] = False
        encode_u_setting.optional['positive'] = False
        encode_u_setting.optional['dimension'] = {
            'length': 1, 'time': -1, 'mass': 0
        }

        encode_phi_setting = copy.deepcopy(block_setting)
        encode_phi_setting.name = 'ENCODE_PHI'
        encode_phi_setting.no_grad = True
        encode_phi_setting.coeff = 1.
        encode_phi_setting.bias = False
        encode_phi_setting.residual = False
        encode_phi_setting.nodes = [1, block_setting.nodes[-1]]
        encode_phi_setting.activations = ['identity']

        dirichlet_phi_setting = copy.deepcopy(encode_phi_setting)
        dirichlet_phi_setting.name = 'DIRICHLET_PHI'
        dirichlet_phi_setting.activations = ['identity']
        dirichlet_phi_setting.no_grad = True

        neumann_phi_setting = copy.deepcopy(encode_phi_setting)
        neumann_phi_setting.name = 'NEUMANN_PHI'
        neumann_phi_setting.activations = ['identity']
        neumann_phi_setting.no_grad = True

        encode_nu_setting = copy.deepcopy(block_setting)
        encode_nu_setting.name = 'ENCODE_DIFFUSION'
        encode_nu_setting.coeff = 1.
        encode_nu_setting.nodes = [1] + block_setting.nodes[1:]
        encode_nu_setting.residual = False
        encode_nu_setting.optional['diff'] = False
        encode_nu_setting.optional['positive'] = False
        encode_nu_setting.optional['dimension'] = {
            'length': 2, 'time': -1, 'mass': 0}

        # # Processor
        conv_phi_setting = copy.deepcopy(block_setting)
        conv_phi_setting.name = 'CONV_PHI'
        conv_phi_setting.optional['diff'] = False

        conv_phi_flux_setting = copy.deepcopy(block_setting)
        conv_phi_flux_setting.name = 'CONV_PHI_FLUX'
        conv_phi_flux_setting.optional['diff'] = False
        conv_phi_flux_setting.optional['positive'] = False
        conv_phi_flux_setting.optional['dimension']['length'] += 1
        conv_phi_flux_setting.optional['dimension']['time'] -= 1

        diffusion_grad_setting = copy.deepcopy(block_setting)
        diffusion_grad_setting.name = 'DIFFUSION_GRAD'
        diffusion_grad_setting.optional['diff'] = False
        diffusion_grad_setting.optional['positive'] = False
        diffusion_grad_setting.optional['dimension']['length'] -= 1

        diff_corr_setting = copy.deepcopy(block_setting)
        diff_corr_setting.name = 'DIFF_CORR'
        diff_corr_setting.optional['diff'] = False
        diff_corr_setting.optional['positive'] = False
        diff_corr_setting.optional['dimension']['length'] -= 1

        inv_phi_setting = copy.deepcopy(block_setting)
        inv_phi_setting.coeff = 1.
        inv_phi_setting.name = 'DECODE_PHI'
        inv_phi_setting.no_grad = False
        inv_phi_setting.residual = False
        inv_phi_setting.activations = ['identity']

        # Create models
        encode_phi_mlp = siml.networks.mlp.MLP(encode_phi_setting)
        dirichlet_phi_mlp = \
            siml.networks.share.Share(
                dirichlet_phi_setting, reference_block=encode_phi_mlp)
        neumann_phi_mlp = \
            siml.networks.share.Share(
                neumann_phi_setting, reference_block=encode_phi_mlp)

        inv_phi_mlp = siml.networks.pinv_mlp.PInvMLP(
            inv_phi_setting, reference_block=encode_phi_mlp)

        if self.use_mlp:
            encode_u_mlp = siml.networks.tensor_operations.EquivariantMLP(
                encode_u_setting)

            int_phi_upwind_mlp = siml.networks.mlp.MLP(conv_phi_setting)
            int_phi_both_mlp = siml.networks.mlp.MLP(conv_phi_setting)
            int_phi_center_mlp = siml.networks.mlp.MLP(conv_phi_setting)

            conv_phi_flux_mlp \
                = siml.networks.tensor_operations.EquivariantMLP(
                    conv_phi_flux_setting)
            diffusion_grad_mlp \
                = siml.networks.tensor_operations.EquivariantMLP(
                    diffusion_grad_setting)
            diff_corr_mlp \
                = siml.networks.tensor_operations.EquivariantMLP(
                    diff_corr_setting)

            encode_nu_mlp = siml.networks.tensor_operations.EquivariantMLP(
                encode_nu_setting)

        else:
            encode_u_mlp = siml.networks.tensor_operations.EnSEquivariantMLP(
                encode_u_setting)

            int_phi_upwind_mlp \
                = siml.networks.tensor_operations.EnSEquivariantMLP(
                    conv_phi_setting)
            int_phi_both_mlp \
                = siml.networks.tensor_operations.EnSEquivariantMLP(
                    conv_phi_setting)
            int_phi_center_mlp \
                = siml.networks.tensor_operations.EnSEquivariantMLP(
                    conv_phi_setting)

            conv_phi_flux_mlp \
                = siml.networks.tensor_operations.EnSEquivariantMLP(
                    conv_phi_flux_setting)
            diffusion_grad_mlp \
                = siml.networks.tensor_operations.EnSEquivariantMLP(
                    diffusion_grad_setting)
            diff_corr_mlp \
                = siml.networks.tensor_operations.EnSEquivariantMLP(
                    diff_corr_setting)

            encode_nu_mlp \
                = siml.networks.tensor_operations.EnSEquivariantMLP(
                    encode_nu_setting)

        dict_mlp = torch.nn.ModuleDict({
            # Encoders
            f"{self.mode}/encode_u_mlp": encode_u_mlp,

            f"{self.mode}/encode_phi_mlp": encode_phi_mlp,
            f"{self.mode}/dirichlet_phi_mlp": dirichlet_phi_mlp,
            f"{self.mode}/neumann_phi_mlp": neumann_phi_mlp,

            f"{self.mode}/encode_nu_mlp": encode_nu_mlp,

            # Processors
            f"{self.mode}/interpolation_both": int_phi_both_mlp,
            f"{self.mode}/interpolation_upwind": int_phi_upwind_mlp,
            f"{self.mode}/interpolation_center": int_phi_center_mlp,
            f"{self.mode}/conv_phi_flux_mlp": conv_phi_flux_mlp,
            f"{self.mode}/diffusion_grad_mlp": diffusion_grad_mlp,
            f"{self.mode}/diff_corr_mlp": diff_corr_mlp,

            # Decoders
            f"{self.mode}/inv_phi_mlp": inv_phi_mlp,

        })

        return dict_mlp

    def solve_mixture(
            self, t_max, delta_t, torch_fv,
            cell_initial_u, cell_initial_p, cell_initial_alpha,
            global_nu_solute, global_nu_solvent,
            global_rho_solute, global_rho_solvent,
            facet_diffusion_alpha, global_gravity,
            facet_dirichlet_u, facet_neumann_u,
            facet_dirichlet_p, facet_neumann_p,
            facet_dirichlet_alpha, facet_neumann_alpha,
            write=False, store=False, evaluate=False,
            output_directory=None, n_forward=None):
        torch_fv.mode = self.mode
        density_scale = (global_rho_solute + global_rho_solvent) / 2
        torch_fv.mass_scale = density_scale * torch_fv.facet_length_scale**3
        torch_fv.time_scale = torch_tensor(np.atleast_2d(delta_t))
        torch_fv.poisson_solver.time_scale = torch_fv.time_scale
        torch_fv.poisson_solver.trainable = False
        torch_fv.center_data = self.center_data
        torch_fv.ml_interpolation = self.ml_interpolation

        if self.deep_processor:
            torch_fv.deep_processor = True
            torch_fv.time_evolution_method = 'explicit'
            torch_fv.n_time_repeat = self.n_repeat_deep_processor
            torch_fv.n_alpha_repeat = 1
        torch_fv.encoded_pressure = self.encoded_pressure
        torch_fv.n_bundle = self.n_bundle
        torch_fv.scale_encdec = self.scale_encdec
        torch_fv.l2_for_scale_encdec = self.l2_for_scale_encdec

        dict_inputs = {
            'u': cell_initial_u, 'p': cell_initial_p,
            'alpha': cell_initial_alpha}

        if n_forward is None:
            n_forward = self.n_forward

        if write or store or evaluate or n_forward < 2:
            dict_results = self._solve_mixture(
                t_max, delta_t, torch_fv,
                cell_initial_u, cell_initial_p, cell_initial_alpha,
                global_nu_solute, global_nu_solvent,
                global_rho_solute, global_rho_solvent,
                facet_diffusion_alpha, global_gravity,
                facet_dirichlet_u, facet_neumann_u,
                facet_dirichlet_p, facet_neumann_p,
                facet_dirichlet_alpha, facet_neumann_alpha,
                write=write, store=store,
                dict_inputs=dict_inputs,
                output_directory=output_directory)

        else:
            n_pushforward = n_forward - 1
            # Pushforward trick
            cell_intermediate_u = cell_initial_u
            cell_intermediate_p = cell_initial_p
            cell_intermediate_alpha = cell_initial_alpha
            if self.encoded_pushforward:
                encode = True
                decode = False
            else:
                encode = True
                decode = True
            with torch.no_grad():
                for _ in range(n_pushforward):
                    # print(f"Pushforward {_ + 1}")
                    dict_results \
                        = self._solve_mixture(
                            t_max / n_forward, delta_t, torch_fv,
                            cell_intermediate_u, cell_intermediate_p,
                            cell_intermediate_alpha,
                            global_nu_solute, global_nu_solvent,
                            global_rho_solute, global_rho_solvent,
                            facet_diffusion_alpha, global_gravity,
                            facet_dirichlet_u, facet_neumann_u,
                            facet_dirichlet_p, facet_neumann_p,
                            facet_dirichlet_alpha, facet_neumann_alpha,
                            encode=encode, decode=decode,
                            write=write, store=store,
                            dict_inputs=dict_inputs,
                            output_directory=output_directory)
                    cell_intermediate_u = dict_results['u'][-1]
                    cell_intermediate_p = dict_results['p'][-1]
                    cell_intermediate_alpha = dict_results['alpha'][-1]
                    if self.encoded_pushforward:
                        encode = False

            # print('After pushforward')
            if self.encoded_pushforward:
                decode = True
            dict_results = self._solve_mixture(
                t_max / n_forward, delta_t, torch_fv,
                cell_intermediate_u, cell_intermediate_p,
                cell_intermediate_alpha,
                global_nu_solute, global_nu_solvent,
                global_rho_solute, global_rho_solvent,
                facet_diffusion_alpha, global_gravity,
                facet_dirichlet_u, facet_neumann_u,
                facet_dirichlet_p, facet_neumann_p,
                facet_dirichlet_alpha, facet_neumann_alpha,
                encode=encode, decode=decode,
                write=write, store=store,
                dict_inputs=dict_inputs,
                output_directory=output_directory)

        return dict_results

    def _solve_mixture(
            self, t_max, delta_t, torch_fv,
            cell_initial_u, cell_initial_p, cell_initial_alpha,
            global_nu_solute, global_nu_solvent,
            global_rho_solute, global_rho_solvent,
            facet_diffusion_alpha, global_gravity,
            facet_dirichlet_u, facet_neumann_u,
            facet_dirichlet_p, facet_neumann_p,
            facet_dirichlet_alpha, facet_neumann_alpha,
            encode=True, decode=True,
            write=False, store=False,
            dict_inputs=None,
            output_directory=None):

        if encode:
            cell_emb_u = torch_fv.apply_mlp(
                self.dict_mlp, 'encode_u_mlp', cell_initial_u,
                reduce_op=cat_tail, scale=self.scale_encdec,
                l2=self.l2_for_scale_encdec)
            if self.encoded_pressure:
                cell_emb_p = torch_fv.apply_mlp(
                    self.dict_mlp, 'encode_p_mlp', cell_initial_p,
                    reduce_op=cat_tail, scale=self.scale_encdec,
                    l2=self.l2_for_scale_encdec)
            else:
                cell_emb_p = torch.cat([
                    cell_initial_p] * self.n_bundle, dim=-1)
            cell_emb_alpha = torch_fv.apply_mlp(
                self.dict_mlp, 'encode_alpha_mlp', cell_initial_alpha,
                reduce_op=cat_tail, scale=self.scale_encdec,
                l2=self.l2_for_scale_encdec)
        else:
            cell_emb_u = cell_initial_u
            cell_emb_p = cell_initial_p
            cell_emb_alpha = cell_initial_alpha

        facet_emb_dirichlet_u = torch_fv.apply_mlp(
            self.dict_mlp, 'dirichlet_u_mlp', facet_dirichlet_u,
            reduce_op=cat_tail, scale=self.scale_encdec,
            l2=self.l2_for_scale_encdec)
        facet_emb_neumann_u = torch_fv.apply_mlp(
            self.dict_mlp, 'neumann_u_mlp', facet_neumann_u,
            reduce_op=cat_tail, scale=self.scale_encdec,
            l2=self.l2_for_scale_encdec)

        facet_emb_dirichlet_p = torch_fv.apply_mlp(
            self.dict_mlp, 'dirichlet_p_mlp', facet_dirichlet_p,
            reduce_op=cat_tail, scale=self.scale_encdec,
            l2=self.l2_for_scale_encdec)
        facet_emb_neumann_p = torch_fv.apply_mlp(
            self.dict_mlp, 'neumann_p_mlp', facet_neumann_p,
            reduce_op=cat_tail, scale=self.scale_encdec,
            l2=self.l2_for_scale_encdec)

        facet_emb_dirichlet_alpha = torch_fv.apply_mlp(
            self.dict_mlp, 'dirichlet_alpha_mlp', facet_dirichlet_alpha,
            reduce_op=cat_tail, scale=self.scale_encdec,
            l2=self.l2_for_scale_encdec)
        facet_emb_neumann_alpha = torch_fv.apply_mlp(
            self.dict_mlp, 'neumann_alpha_mlp', facet_neumann_alpha,
            reduce_op=cat_tail, scale=self.scale_encdec,
            l2=self.l2_for_scale_encdec)

        global_emb_nu_solute = torch_fv.apply_mlp(
            self.dict_mlp, 'encode_nu_mlp', global_nu_solute)
        global_emb_nu_solvent = torch_fv.apply_mlp(
            self.dict_mlp, 'encode_nu_mlp', global_nu_solvent)
        global_emb_rho_solute = torch_fv.apply_mlp(
            self.dict_mlp, 'encode_rho_mlp', global_rho_solute)
        global_emb_rho_solvent = torch_fv.apply_mlp(
            self.dict_mlp, 'encode_rho_mlp', global_rho_solvent)
        global_emb_gravity = torch_fv.apply_mlp(
            self.dict_mlp, 'encode_gravity_mlp', global_gravity)
        facet_emb_diffusion_alpha = torch_fv.apply_mlp(
            self.dict_mlp, 'encode_diffusion_alpha_mlp', facet_diffusion_alpha)

        if not self.encoded_pressure:
            facet_emb_dirichlet_p = torch.cat([
                facet_emb_dirichlet_p] * self.n_bundle, dim=-1)
            facet_emb_neumann_p = torch.cat([
                facet_emb_neumann_p] * self.n_bundle, dim=-1)

        dict_results = torch_fv.solve_mixture(
            t_max, delta_t * self.n_bundle,
            cell_emb_u, cell_emb_p, cell_emb_alpha,
            facet_emb_dirichlet_u, facet_emb_neumann_u,
            facet_emb_dirichlet_p, facet_emb_neumann_p,
            facet_dirichlet_alpha=facet_emb_dirichlet_alpha,
            facet_neumann_alpha=facet_emb_neumann_alpha,
            global_nu_solute=global_emb_nu_solute,
            global_nu_solvent=global_emb_nu_solvent,
            global_rho_solute=global_emb_rho_solute,
            global_rho_solvent=global_emb_rho_solvent,
            facet_diffusion_alpha=facet_emb_diffusion_alpha,
            global_gravity=global_emb_gravity,
            facet_original_dirichlet_u=facet_dirichlet_u,
            facet_original_neumann_u=facet_neumann_u,
            facet_original_dirichlet_p=facet_dirichlet_p,
            facet_original_neumann_p=facet_neumann_p,
            dict_mlp=self.dict_mlp, write=write, store=store,
            output_directory=output_directory)
        if self.fv_residual:
            torch_fv.trainable = False
            dict_linear_results = torch_fv.solve_mixture(
                t_max, delta_t * self.n_bundle,
                cell_emb_u, cell_emb_p, cell_emb_alpha,
                facet_emb_dirichlet_u, facet_emb_neumann_u,
                facet_emb_dirichlet_p, facet_emb_neumann_p,
                facet_dirichlet_alpha=facet_emb_dirichlet_alpha,
                facet_neumann_alpha=facet_emb_neumann_alpha,
                global_nu_solute=global_emb_nu_solute,
                global_nu_solvent=global_emb_nu_solvent,
                global_rho_solute=global_emb_rho_solute,
                global_rho_solvent=global_emb_rho_solvent,
                facet_diffusion_alpha=facet_emb_diffusion_alpha,
                global_gravity=global_emb_gravity,
                facet_original_dirichlet_u=facet_dirichlet_u,
                facet_original_neumann_u=facet_neumann_u,
                facet_original_dirichlet_p=facet_dirichlet_p,
                facet_original_neumann_p=facet_neumann_p,
                dict_mlp=self.dict_mlp, write=write, store=store,
                output_directory=output_directory)
            torch_fv.trainable = True

            dict_results = {
                k: self.fv_residual_nonlinear_weight * dict_results[k]
                + (1 - self.fv_residual_nonlinear_weight)
                * dict_linear_results[k]
                for k in dict_results.keys()}

        if decode:
            if self.temporal_bundling:
                dict_results['u'] = einops.rearrange(
                    torch_fv.apply_mlp(
                        self.dict_mlp, 'inv_u_mlp', dict_results['u'],
                        reduce_op=torch.stack,
                        n_split=self.n_bundle, scale=self.scale_encdec,
                        l2=self.l2_for_scale_encdec),
                    't s ... -> (s t) ...')
                dict_results['p'] = einops.rearrange(
                    torch_fv.apply_mlp(
                        self.dict_mlp, 'inv_p_mlp', dict_results['p'],
                        reduce_op=torch.stack,
                        n_split=self.n_bundle, scale=self.scale_encdec,
                        l2=self.l2_for_scale_encdec),
                    't s ... -> (s t) ...')
                dict_results['alpha'] = einops.rearrange(
                    torch_fv.apply_mlp(
                        self.dict_mlp, 'inv_alpha_mlp', dict_results['alpha'],
                        reduce_op=torch.stack,
                        n_split=self.n_bundle, scale=self.scale_encdec,
                        l2=self.l2_for_scale_encdec),
                    't s ... -> (s t) ...')
            else:
                dict_results['u'] = torch_fv.apply_mlp(
                    self.dict_mlp, 'inv_u_mlp', dict_results['u'])
                dict_results['p'] = torch_fv.apply_mlp(
                    self.dict_mlp, 'inv_p_mlp', dict_results['p'])
                dict_results['alpha'] = torch_fv.apply_mlp(
                    self.dict_mlp, 'inv_alpha_mlp', dict_results['alpha'])

            if self.time_average:
                for i_average in range(self.time_average_depth):
                    dict_results = self.apply_time_average(
                        i_average, dict_inputs, dict_results)

        return dict_results

    def apply_time_average(self, i_average, dict_inputs, dict_results):
        dict_ts = {
            key:
            self._compute_time_average(i_average, key, torch.cat(
                [dict_inputs[key][None, ...], dict_results[key]], dim=0))
            for key in dict_results.keys()}
        return dict_ts

    def _compute_time_average(self, i_average, key, ts):
        if self.trainable_bundle:
            coeff = torch.sigmoid(
                self.dict_mlp[
                    f"{self.mode}{i_average}/tb_{key}"].linears[0].weight) \
                * (1 - self.tb_base) + self.tb_base
            return ts[1:] * coeff + ts[:-1] * (1 - coeff)
        else:
            return ts[1:] * self.time_average_coeff \
                + ts[:-1] * (1 - self.time_average_coeff)

    def solve_convection_diffusion(
            self, t_max, delta_t, torch_fv,
            cell_initial_phi, facet_velocity, facet_diffusion,
            facet_dirichlet, facet_neumann, periodic=None,
            write=False, store=False, evaluate=False, output_directory=None):
        torch_fv.mode = self.mode
        torch_fv.mass_scale = None
        torch_fv.time_scale = torch_tensor(np.atleast_2d(delta_t))
        torch_fv.center_data = self.center_data
        torch_fv.ml_interpolation = self.ml_interpolation

        if self.deep_processor:
            torch_fv.deep_processor = True
            torch_fv.n_time_repeat = self.n_repeat_deep_processor
            torch_fv.n_alpha_repeat = 1
        torch_fv.encoded_pressure = self.encoded_pressure
        torch_fv.n_bundle = self.n_bundle
        torch_fv.scale_encdec = self.scale_encdec
        torch_fv.l2_for_scale_encdec = self.l2_for_scale_encdec

        if write or store or evaluate or self.n_forward < 2:
            dict_results = self._solve_convection_diffusion(
                t_max, delta_t, torch_fv,
                cell_initial_phi, facet_velocity, facet_diffusion,
                facet_dirichlet, facet_neumann, periodic=periodic,
                write=write, store=store, output_directory=output_directory)

        else:
            n_pushforward = self.n_forward - 1
            # Pushforward trick
            if self.encoded_pushforward:
                encode = True
                decode = False
            else:
                encode = True
                decode = True
            cell_intermediate_phi = cell_initial_phi
            with torch.no_grad():
                for _ in range(n_pushforward):
                    # print(f"Pushforward {_ + 1}")
                    dict_results = self._solve_convection_diffusion(
                        t_max / self.n_forward, delta_t, torch_fv,
                        cell_intermediate_phi, facet_velocity, facet_diffusion,
                        facet_dirichlet, facet_neumann, periodic=periodic,
                        write=write, store=store,
                        encode=encode, decode=decode,
                        output_directory=output_directory)
                    cell_intermediate_phi = dict_results['phi'][-1]
                    if self.encoded_pushforward:
                        encode = False
            # print('After pushforward')
            if self.encoded_pushforward:
                decode = True
            dict_results = self._solve_convection_diffusion(
                t_max / self.n_forward, delta_t, torch_fv,
                cell_intermediate_phi, facet_velocity, facet_diffusion,
                facet_dirichlet, facet_neumann, periodic=periodic,
                encode=encode, decode=decode,
                write=write, store=store, output_directory=output_directory)

        return dict_results

    def _solve_convection_diffusion(
            self, t_max, delta_t, torch_fv,
            cell_initial_phi, facet_velocity, facet_diffusion,
            facet_dirichlet, facet_neumann, periodic=None,
            encode=True, decode=True,
            write=False, store=False, output_directory=None):

        if encode:
            cell_emb_phi = torch_fv.apply_mlp(
                self.dict_mlp, 'encode_phi_mlp', cell_initial_phi)
        else:
            cell_emb_phi = cell_initial_phi

        facet_emb_velocity = torch_fv.apply_mlp(
            self.dict_mlp, 'encode_u_mlp', facet_velocity)
        facet_emb_diffusion = torch_fv.apply_mlp(
            self.dict_mlp, 'encode_nu_mlp', facet_diffusion)
        facet_emb_dirichlet = torch_fv.apply_mlp(
            self.dict_mlp, 'dirichlet_phi_mlp', facet_dirichlet)
        facet_emb_neumann = torch_fv.apply_mlp(
            self.dict_mlp, 'neumann_phi_mlp', facet_neumann)

        dict_results = torch_fv.solve_convection_diffusion(
            t_max, delta_t,
            cell_emb_phi, facet_emb_velocity, facet_emb_diffusion,
            facet_emb_dirichlet, facet_emb_neumann, dict_mlp=self.dict_mlp,
            periodic=periodic,
            write=write, store=store, output_directory=output_directory)

        if decode:
            dict_results['phi'] = torch_fv.apply_mlp(
                self.dict_mlp, 'inv_phi_mlp', dict_results['phi'])
        return dict_results
