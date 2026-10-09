import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x_transformer import (
    AE_style,
    AE_style_jokerParam,
    Decoder_no_dtime_style,
    Decoder_no_dtime_style_jokerParam,
    Encoder_no_dtime,
    StyleEncoder,
)


COMMON = {
    'max_seq_len': 16,
    'dim': 32,
    'depth': 1,
    'heads': 4,
    'rotary_pos_emb': True,
    'attn_flash': False,
}

CFG = {
    'num_buttons': 19,
    'joker_ratio': 0.25,
    'loss_recons': 1.0,
    'loss_margin': 0.0,
    'loss_contour': 0.0,
    'loss_deviate': 0.0,
}


def model_parts(*, joker: bool):
    decoder_cls = (
        Decoder_no_dtime_style_jokerParam if joker
        else Decoder_no_dtime_style
    )
    return {
        'decoder': decoder_cls(**COMMON),
        'encoder': Encoder_no_dtime(**COMMON),
        'style_encoder': StyleEncoder(
            max_seq_len=8,
            dim=32,
            depth=1,
            heads=4,
            rotary_pos_emb=True,
            attn_flash=False,
        ),
    }


class StyleJokerParamTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_warm_start_is_exact_and_only_adds_zero_joker_parameter(self):
        base = AE_style(cfg=CFG, **model_parts(joker=False)).eval()
        model = AE_style_jokerParam(cfg=CFG, **model_parts(joker=True)).eval()

        status = model.load_state_dict(base.state_dict(), strict=False)
        self.assertEqual(status.missing_keys, ['decoder.joker_mode'])
        self.assertEqual(status.unexpected_keys, [])
        self.assertEqual(model.decoder.joker_mode.count_nonzero().item(), 0)

        batch = {
            'pitch': torch.randint(0, 128, (2, 9)),
            'style_pitch': torch.randint(0, 128, (2, 8)),
            'style_mask': torch.ones(2, 8, dtype=torch.bool),
        }
        with torch.no_grad():
            base_loss, base_acc = base(batch)
            model_loss, model_acc = model(batch)

        self.assertTrue(torch.equal(base_loss['loss_total'], model_loss['loss_total']))
        self.assertTrue(torch.equal(base_acc, model_acc))
        self.assertEqual(model_loss['joker_fraction'].item(), 0.0)

    def test_all_positions_are_eligible_and_joker_parameter_gets_gradient(self):
        model = AE_style_jokerParam(cfg=CFG, **model_parts(joker=True)).train()
        mask = model.make_joker_mask(2, 8, torch.device('cpu'), deterministic=True)

        # There is no channel filter: each sequence gets exactly 25% joker notes.
        self.assertEqual(mask.sum(dim=1).tolist(), [2, 2])

        batch = {
            'pitch': torch.randint(0, 128, (2, 9)),
            'style_pitch': torch.randint(0, 128, (2, 8)),
            'style_mask': torch.ones(2, 8, dtype=torch.bool),
            'joker_mask': mask,
        }
        loss, _ = model(batch)
        loss['loss_total'].backward()

        gradient = model.decoder.joker_mode.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0.0)
        self.assertAlmostEqual(loss['joker_fraction'].item(), 0.25)

    def test_discrete_joker_uses_zero_scalar_and_separate_mask(self):
        model = AE_style_jokerParam(cfg=CFG, **model_parts(joker=True))
        real, joker_mask = model._buttons_discrete_to_inputs(
            torch.tensor([[0, 9, 18, 19]])
        )

        self.assertEqual(joker_mask.tolist(), [[False, False, False, True]])
        self.assertTrue(torch.equal(real, torch.tensor([[-1.0, 0.0, 1.0, 0.0]])))


if __name__ == '__main__':
    unittest.main()
