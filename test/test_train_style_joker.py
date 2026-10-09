import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_style import (
    StyleMusicSamplerDataset,
    encoder_outside_percentage,
    make_joker_training_mask,
)


def sample_data():
    boundary = [126, 126, 0, 0, 0]
    events = []
    for offset in (0, 12):
        events.append(boundary)
        events.extend(
            [0 if i % 3 else 2, 4, 48 + (i + offset) % 36, 80, 0]
            for i in range(96)
        )
    return torch.tensor(events, dtype=torch.long).flatten()


class JokerTrainingDataTest(unittest.TestCase):
    def setUp(self):
        self.cfg = {'model_type': 'AE_style_jokerParam'}

    def test_training_masks_include_guided_long_spans_and_full_joker(self):
        torch.manual_seed(7)
        mask = make_joker_training_mask(4096, 512, torch.device('cpu'))
        counts = mask.sum(dim=1)
        guided = counts == 0
        full = counts == 512
        mixed = ~(guided | full)

        self.assertEqual(mask.shape, (4096, 512))
        self.assertLess(abs(guided.float().mean().item() - 0.25), 0.03)
        self.assertLess(abs(mixed.float().mean().item() - 0.50), 0.03)
        self.assertLess(abs(full.float().mean().item() - 0.25), 0.03)
        self.assertTrue(((counts[mixed] >= 32) & (counts[mixed] <= 128)).all().item())

        # A mixed sequence must contain exactly one consecutive passage.
        starts = mask[:, 0].long().sum()
        starts += (mask[:, 1:] & ~mask[:, :-1]).long().sum()
        self.assertEqual(starts.item(), (mixed | full).sum().item())

    def test_joker_validation_is_repeatable(self):
        dataset = StyleMusicSamplerDataset(
            sample_data(), 8, 8, is_eval=True, cfg=self.cfg
        )
        random.seed(1)
        torch.manual_seed(1)
        first = dataset[3]
        random.seed(999)
        torch.manual_seed(999)
        again = dataset[3]
        for key in ('pitch', 'style_pitch', 'style_mask',
                    'harm_regime', 'harm_strength'):
            self.assertTrue(torch.equal(first[key], again[key]), key)

    def test_joker_target_keeps_source_chord_note_order(self):
        dataset = StyleMusicSamplerDataset(
            sample_data(), 3, 3, cfg=self.cfg
        )
        events = torch.tensor([
            [0, 4, 60, 80, 0],
            [0, 4, 64, 80, 0],
            [0, 4, 67, 80, 0],
            [2, 4, 72, 80, 0],
        ])
        for seed in range(10):
            random.seed(seed)
            torch.manual_seed(seed)
            self.assertEqual(
                dataset._process_target_window(events, 0)['pitch'].tolist(),
                [60, 64, 67, 72],
            )

    def test_encoder_outside_percentage_excludes_padding(self):
        class FixedEncoder(torch.nn.Module):
            def forward(self, note_tokens):
                return torch.tensor(
                    [[-1.2, -1.0, 0.0, 1.0, 1.2]],
                    device=note_tokens['pitch'].device,
                )

        model = SimpleNamespace(
            encoder=FixedEncoder(),
            ignore_index=128,
        )
        pitch = torch.tensor([[50, 60, 61, 62, 63, 128]])
        self.assertAlmostEqual(encoder_outside_percentage(model, pitch), 25.0)


if __name__ == '__main__':
    unittest.main()
