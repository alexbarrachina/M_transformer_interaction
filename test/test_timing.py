"""Correctness checks and short training smoke runs; no quality evaluation."""

import ast
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
from typing import List, Tuple
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_style
from model_loader import load_model
from models import get_model_hparams
from params import HARMONY_CHANNEL
from timing import LiveTimingContext, TimingHistory, TIME_UNIT, augment_timing, timing_features
from train_style import (
    StyleMusicSamplerDataset, load_checkpoint, make_timing_optimizer,
    save_timing_checkpoint, set_timing_stage, warm_start_timing,
)


def small_config():
    cfg = get_model_hparams('AE_style_jokerParam_dtime_tester_v1')
    cfg.update(emb_dim=32, num_layers=4, heads=4, seq_len=8, style_seq_len=8,
               style_encoder_depth=1, batch_size=2, num_workers=0, use_logs=False,
               timing_warmup_steps=2,
               print_stats_every=1, epochs=2)
    return cfg


def piece():
    return torch.tensor([[0 if i % 3 else 8, 4, 48 + i % 36, 80, 0]
                         for i in range(96)], dtype=torch.long)


def training_batch():
    features, valid = timing_features([None, .25, 0, .125, .5, .25, .1, .2, .3])
    return {
        'pitch': torch.tensor([[60, 64, 67, 65, 62, 60, 64, 67, 72]]),
        'style_pitch': torch.tensor([[48, 52, 55, 60, 64, 67, 72, 76]]),
        'style_mask': torch.ones(1, 8, dtype=torch.bool),
        'joker_mask': torch.tensor([[False, True, False, True, False, True, False, True]]),
        'timing_features': features.unsqueeze(0), 'timing_mask': valid.unsqueeze(0),
    }


class TimingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        random.seed(7)

    def test_features_are_causal_and_streaming_matches_batch(self):
        intervals = [None, 0, .032, .125, .25, .5, 8.0, float('nan'), -1] + [.2] * 20
        expected, valid = timing_features(intervals)
        stream = TimingHistory()
        for i, interval in enumerate(intervals):
            row, known = stream.push(interval)
            torch.testing.assert_close(torch.tensor(row), expected[i])
            self.assertEqual(known, valid[i])
            prefix, _ = timing_features(intervals[:i + 1])
            self.assertTrue(torch.equal(prefix, expected[:i + 1]))
        self.assertTrue(torch.isfinite(expected).all())
        self.assertEqual(expected[1, 3], 1)
        self.assertTrue(valid[1])
        self.assertFalse(valid[0])
        self.assertAlmostEqual(expected[2, 0].item(), 1 / 7)
        self.assertEqual(expected[6, 0], 1)
        for ratio in (.5, 1, 2):
            history = TimingHistory()
            history.push(.25)
            self.assertAlmostEqual(history.push(.25 * ratio)[0][1], math.log2(ratio) / 3)
        self.assertAlmostEqual(stream.scale, .2)

    def test_crop_uses_preceding_notes_and_accumulates_marker_time(self):
        events = piece()
        events = torch.cat([events[:30], torch.tensor([[5, 0, 2, 0, HARMONY_CHANNEL]]), events[30:]])
        dataset = StyleMusicSamplerDataset(events.flatten(), 8, 8, is_eval=True, cfg=small_config())
        note_positions = torch.where(events[:, 4] != HARMONY_CHANNEL)[0]
        onsets = events[:, 0].double().cumsum(0)[note_positions] * TIME_UNIT
        complete, known = timing_features([None] + torch.diff(onsets).tolist())
        crop = dataset._target_timing(events, 31)
        torch.testing.assert_close(crop['timing_features'], complete[30:39])
        self.assertTrue(torch.equal(crop['timing_mask'], known[30:39]))
        self.assertAlmostEqual(crop['timing_features'][0, 0].item(), math.log1p(13) / math.log(128))
        first = dataset[0]
        again = dataset[0]
        self.assertTrue(all(torch.equal(first[k], again[k]) for k in first))

    def test_augmentation_preserves_unknowns_and_exercises_pause_cap(self):
        original = [None] + [.25, 0, .125, .5] * 8
        saw_pause = saw_spread = False
        for seed in range(100):
            transformed = augment_timing(original, 17, random.Random(seed))
            self.assertIsNone(transformed[0])
            self.assertEqual(len(transformed), len(original))
            self.assertTrue(all(d >= 0 for d in transformed[1:]))
            saw_pause |= max(transformed[1:]) > 4.064
            saw_spread |= any(d == 0 and t > 0 for d, t in zip(original[1:], transformed[1:]))
        self.assertTrue(saw_pause and saw_spread)
        self.assertEqual(original, [None] + [.25, 0, .125, .5] * 8)

    def test_live_history_survives_eviction_toggle_and_restarts(self):
        live = LiveTimingContext([60, 62, 64, 65], [0, 1, 2, 3], [0, 8, 8, 8], 4)
        timestamps = [1 + i * .125 for i in range(20)]
        history = TimingHistory()
        for interval in [None, .256, .256, .256]:
            history.push(interval)
        for i, timestamp in enumerate(timestamps):
            live.enabled = i != 5
            context, restarted = live.begin_note(timestamp, 3)
            expected, _ = history.push(None if i == 0 else .125)
            torch.testing.assert_close(context['timing_features'][0, -1], torch.tensor(expected))
            self.assertFalse(restarted)
            if i == 5:
                self.assertFalse(context['timing_mask'].any())
            live.finish_note(70)
        self.assertEqual(len(live.pitches), 5)
        before = list(live.features)
        live.enabled = False
        self.assertEqual(list(live.features), before)
        context, restarted = live.begin_note(timestamps[-1] + 10, 2)
        self.assertTrue(restarted)
        self.assertFalse(live.enabled)
        self.assertEqual(context['pitch'][0, :-1].tolist(), [60, 62, 64, 65])
        self.assertFalse(live.valid[-1])

    def test_zero_initialization_legacy_loading_and_training_alignment(self):
        cfg = small_config()
        base = load_model(cfg={**cfg, 'timing_enabled': False}, set_only=True).eval()
        model = load_model(cfg=cfg, set_only=True).eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'base.pth'
            torch.save(base.state_dict(), path)
            warm_start_timing(model, path)
            self.assertEqual(len(model.timing_base_checkpoint['sha256']), 64)
            legacy = load_model(cfg={**cfg, 'timing_enabled': False, 'ckpt_file_name': str(path)},
                                compile_mode='none').eval()
            incomplete = dict(base.state_dict())
            del incomplete['decoder.input_proj.weight']
            torch.save(incomplete, path)
            with self.assertRaisesRegex(ValueError, 'Unsafe timing warm start'):
                warm_start_timing(model, path)
        batch = training_batch()
        with torch.no_grad():
            for joker in (False, True):
                batch['joker_mask'].fill_(joker)
                recorded = []
                hook = model.decoder.register_forward_pre_hook(lambda _, args: recorded.append(args[0]))
                timed_loss, _ = model(batch)
                hook.remove()
                base_loss, _ = base(batch)
                legacy_loss, _ = legacy(batch)
                torch.testing.assert_close(timed_loss['loss_total'], base_loss['loss_total'], rtol=0, atol=0)
                torch.testing.assert_close(legacy_loss['loss_total'], base_loss['loss_total'], rtol=0, atol=0)
                self.assertTrue(torch.equal(recorded[0]['timing_features'], batch['timing_features'][:, 1:]))
                self.assertTrue(torch.equal(recorded[0]['pitch'], batch['pitch'][:, :-1]))
        # A learned branch must still vanish when masked, including its bias.
        torch.nn.init.normal_(model.decoder.timing_mlp[-1].weight, std=.02)
        torch.nn.init.ones_(model.decoder.timing_mlp[-1].bias)
        batch['timing_mask'].zero_()
        batch['timing_features'].fill_(float('nan'))
        torch.testing.assert_close(model(batch)[0]['loss_total'], base(batch)[0]['loss_total'])

    def test_padding_does_not_create_notes_or_invalid_embedding_indices(self):
        cfg = small_config()
        dataset = StyleMusicSamplerDataset(piece().flatten(), 8, 8, is_eval=True, cfg=cfg)
        batch = training_batch()
        short = dataset._process_target_window(piece()[:3], 0)
        batch['pitch'] = short['pitch'].unsqueeze(0)
        timing = dataset._target_timing(piece()[:3], 0)
        batch.update({k: v.unsqueeze(0) for k, v in timing.items()})
        self.assertEqual(batch['pitch'][0, 3:].tolist(), [128] * 6)
        self.assertFalse(batch['timing_mask'][0, 3:].any())
        model = load_model(cfg=cfg, set_only=True)
        loss, _ = model(batch)
        self.assertTrue(torch.isfinite(loss['loss_recons']))
        loss['loss_recons'].backward()

    def test_cached_predictions_match_full_history_with_learned_timing(self):
        model = load_model(cfg=small_config(), set_only=True).eval()
        torch.nn.init.normal_(model.decoder.timing_mlp[-1].weight, std=.02)
        batch = training_batch()
        style = model.encode_style(batch['style_pitch'], batch['style_mask'])
        cache = None
        captured = []
        hook = model.decoder.register_forward_hook(lambda _, args, output: captured.append(output[0][:, -1].clone()))
        for length in range(2, 9):
            context = {k: batch[k][:, :length] for k in ('pitch', 'timing_features', 'timing_mask')}
            context['button'] = torch.tensor([[0, 19, 3, 4, 19, 6, 7, 8]])[:, :length]
            _, cache = model.gen_pitch_token(context, style, batch['style_mask'], cache=cache)
            model.gen_pitch_token(context, style, batch['style_mask'])
            torch.testing.assert_close(captured[-2], captured[-1], rtol=1e-4, atol=1e-5)
        hook.remove()

    def test_both_stages_freeze_expected_parameters_and_resume(self):
        cfg = small_config()
        with tempfile.TemporaryDirectory() as folder:
            cfg['ckpt_file_name'] = str(Path(folder) / 'latest.pth')
            model = load_model(cfg=cfg, set_only=True)
            model.timing_base_checkpoint = {'path': 'test', 'sha256': 'test'}
            optimizer = make_timing_optimizer(model)
            initial = {k: v.clone() for k, v in model.named_parameters()}
            batch = training_batch()
            for step in range(2):
                model.train()
                set_timing_stage(model, optimizer, step)
                self.assertFalse(model.encoder.training)
                optimizer.zero_grad()
                model(batch)[0]['loss_recons'].backward()
                optimizer.step()
            changed = [k for k, p in model.named_parameters() if not torch.equal(initial[k], p)]
            self.assertTrue(changed)
            self.assertTrue(all(k.startswith('decoder.timing_mlp.') for k in changed))
            self.assertGreater(model.decoder.timing_mlp[0].weight.grad.abs().sum(), 0)
            save_timing_checkpoint(model, optimizer, 0, 2)
            expected_random = (random.random(), torch.rand(2))
            resumed = load_model(cfg=cfg, set_only=True)
            resumed_optimizer = make_timing_optimizer(resumed)
            self.assertEqual(load_checkpoint(resumed, resumed_optimizer, cfg['ckpt_file_name'], 'cpu'), (0, 2))
            self.assertEqual(random.random(), expected_random[0])
            self.assertTrue(torch.equal(torch.rand(2), expected_random[1]))
            before = {k: v.clone() for k, v in resumed.named_parameters()}
            resumed_optimizer.zero_grad()
            resumed(batch)[0]['loss_recons'].backward()
            for name, p in resumed.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                else:
                    self.assertIsNone(p.grad, name)
            resumed_optimizer.step()
            changed = [k for k, p in resumed.named_parameters() if not torch.equal(before[k], p)]
            self.assertTrue(any('to_logits' in k for k in changed))
            for name, p in resumed.named_parameters():
                if name.startswith(('encoder.', 'style_encoder.', 'decoder.pitch_emb.', 'decoder.input_proj.')):
                    self.assertTrue(torch.equal(initial[name], p), name)
            for index, kind in enumerate(resumed.decoder.attn_layers.layer_types):
                for p in resumed.decoder.attn_layers.layers[index].parameters():
                    self.assertEqual(p.requires_grad, index >= 6 and kind in ('a', 'f'))

    def test_training_entrypoint_smoke(self):
        cfg = small_config()
        with tempfile.TemporaryDirectory() as folder:
            cfg['ckpt_file_name'] = str(Path(folder) / 'latest.pth')
            cfg['init_from_ckpt'] = str(Path(folder) / 'base.pth')
            log_path = Path(folder) / 'logs' / 'training.log'
            base = load_model(cfg={**cfg, 'timing_enabled': False}, set_only=True)
            torch.save(base.state_dict(), cfg['init_from_ckpt'])
            with patch.object(train_style, 'get_model_hparams', return_value=cfg), \
                 patch.object(train_style, 'Any_Pickle_File_Reader', return_value=piece().flatten().tolist()) as reader, \
                 patch.object(train_style, 'RESUME', True), \
                 patch.object(train_style, 'count', return_value=range(1)), \
                 patch.dict('os.environ', {'BATCH_SIZE': '2', 'NUM_WORKERS': '0',
                                          'TRAIN_TRACE': '0', 'TRAIN_LOG_FILE': str(log_path)}):
                train_style.main()
                self.assertEqual(reader.call_count, 2)  # training and validation datasets
                saved = torch.load(cfg['ckpt_file_name'], map_location='cpu')
                self.assertEqual(saved['steps'], 5)
                self.assertEqual(saved['stage'], 'adaptation')
                self.assertTrue(all(torch.isfinite(t).all() for t in saved['model_state_dict'].values()))
                self.assertEqual(len(saved['optimizer_state_dict']['state']),
                                 sum(len(g['params']) for g in saved['optimizer_state_dict']['param_groups']))
                restored = load_model(cfg=cfg, compile_mode='none')
                for name, value in restored.state_dict().items():
                    self.assertTrue(torch.equal(value, saved['model_state_dict'][name]), name)
                # Local logs must persist metrics and checkpoint saves even
                # when W&B and detailed diagnostic tracing are both disabled.
                log = log_path.read_text()
                for text in ('Training run started', 'Configuration:', 'Training metrics:',
                             'Validation metrics:',
                             '"train_acc":', '"timing_stage": "warmup"',
                             '"timing_stage": "adaptation"', 'Checkpoint saved:',
                             '"step": 4',
                             str(cfg['ckpt_file_name'])):
                    self.assertIn(text, log)

                # Resume appends to the same log and continues from the saved update.
                train_style.main()
                resumed_log = log_path.read_text()
                self.assertTrue(resumed_log.startswith(log))
                self.assertIn('epoch=0 step=5', resumed_log)
                self.assertEqual(torch.load(cfg['ckpt_file_name'], map_location='cpu')['steps'], 10)

    def test_training_failure_is_written_to_local_log(self):
        with tempfile.TemporaryDirectory() as folder:
            log_path = Path(folder) / 'failure.log'
            with patch.dict('os.environ', {'TRAIN_LOG_FILE': str(log_path)}), \
                 patch.object(train_style, '_train', side_effect=RuntimeError('test training failure')):
                with self.assertRaisesRegex(RuntimeError, 'test training failure'):
                    train_style.main()
            log = log_path.read_text()
            self.assertIn('Training failed', log)
            self.assertIn('Traceback (most recent call last)', log)
            self.assertIn('RuntimeError: test training failure', log)

    def test_unlimited_training_resumes_old_limit_and_keeps_periodic_checkpoint(self):
        cfg = small_config()
        cfg.update(timing_total_steps=10000, epochs=1)
        with tempfile.TemporaryDirectory() as folder:
            cfg['ckpt_file_name'] = str(Path(folder) / 'latest.pth')
            log_path = Path(folder) / 'unlimited.log'
            model = load_model(cfg=cfg, set_only=True)
            model.timing_base_checkpoint = {'path': 'test', 'sha256': 'test'}
            optimizer = make_timing_optimizer(model)
            # Prepare a trained timing branch and an old capped checkpoint.
            for _ in range(cfg['timing_warmup_steps']):
                optimizer.zero_grad()
                model(training_batch())[0]['loss_recons'].backward()
                optimizer.step()
            save_timing_checkpoint(model, optimizer, 0, 10000)
            cfg.update(timing_total_steps=None)

            original_step = torch.optim.Adam.step
            updates = []

            def interrupt_before_seventh_update(optimizer, *args, **kwargs):
                if len(updates) == 6:
                    raise KeyboardInterrupt
                result = original_step(optimizer, *args, **kwargs)
                updates.append(1)
                return result

            with patch.object(train_style, 'get_model_hparams', return_value=cfg), \
                 patch.object(train_style, 'Any_Pickle_File_Reader', return_value=piece().flatten().tolist()), \
                 patch.object(train_style, 'RESUME', True), \
                 patch.object(torch.optim.Adam, 'step', new=interrupt_before_seventh_update), \
                 patch.dict('os.environ', {'BATCH_SIZE': '2', 'NUM_WORKERS': '0',
                                          'TRAIN_TRACE': '0', 'TRAIN_LOG_FILE': str(log_path)}):
                with self.assertRaises(KeyboardInterrupt):
                    train_style.main()

            saved = torch.load(cfg['ckpt_file_name'], map_location='cpu')
            self.assertEqual(saved['steps'], 10005)
            self.assertEqual(saved['epoch'], 0)  # last completed save before Ctrl+C
            self.assertEqual(saved['stage'], 'adaptation')
            self.assertIsNone(saved['cfg']['timing_total_steps'])
            self.assertIsNotNone(saved['scaler_state_dict'])
            self.assertEqual(len(updates), 6)
            self.assertIn('Checkpoint saved:', log_path.read_text())
            self.assertIn('step=10005', log_path.read_text())
            self.assertIn('Epoch started: epoch=1', log_path.read_text())

    def test_timing_training_preserves_legacy_wandb_metrics(self):
        cfg = small_config()
        cfg.update(use_logs=True, validate_every=1)
        train_records = []
        eval_records = []
        real_load_model = train_style.load_model

        def capture_output(model, inputs, output):
            loss, acc = output
            if model.training:
                train_records.append({
                    'loss_margin': loss['loss_margin'].item(),
                    'loss_recons': loss['loss_recons'].item(),
                    'encoder_training': model.encoder.training,
                    'style_training': model.style_encoder.training,
                })
            else:
                eval_records.append({
                    'val_loss': loss['loss_total'].item(),
                    'val_acc': acc.item(),
                    'has_timing': 'timing_features' in inputs[0] and inputs[0]['timing_mask'].any().item(),
                })

        def instrumented_model(*args, **kwargs):
            model = real_load_model(*args, **kwargs)
            model.register_forward_hook(capture_output)
            return model

        with tempfile.TemporaryDirectory() as folder:
            cfg['ckpt_file_name'] = str(Path(folder) / 'latest.pth')
            cfg['init_from_ckpt'] = str(Path(folder) / 'base.pth')
            base = load_model(cfg={**cfg, 'timing_enabled': False}, set_only=True)
            torch.save(base.state_dict(), cfg['init_from_ckpt'])
            wandb_run = SimpleNamespace(log=Mock(), summary={}, finish=Mock())
            wandb = Mock()
            wandb.init.return_value = wandb_run
            with patch.object(train_style, 'get_model_hparams', return_value=cfg), \
                 patch.object(train_style, 'load_model', side_effect=instrumented_model), \
                 patch.object(train_style, 'Any_Pickle_File_Reader', return_value=piece().flatten().tolist()), \
                 patch.object(train_style, 'RESUME', False), \
                 patch.object(train_style, 'count', return_value=range(1)), \
                 patch.dict('sys.modules', {'wandb': wandb}), \
                 patch.dict('os.environ', {'BATCH_SIZE': '2', 'NUM_WORKERS': '0',
                                          'TRAIN_TRACE': '0', 'TRAIN_LOG_FILE': str(Path(folder) / 'metrics.log')}):
                train_style.main()

            logged = [call.args[0] for call in wandb.log.call_args_list]
            keys = {key for row in logged for key in row}
            expected = {
                'loss_total', 'loss_recons', 'train_acc', 'loss_margin', 'loss_deviate',
                'loss_contour_all', 'loss_multi_step', 'val_loss', 'val_acc',
                'val_joker_loss', 'val_joker_acc', 'val_guided_loss', 'val_guided_acc',
                'train_encoder_outside_pct', 'val_encoder_outside_pct',
                'loss_recons_guided', 'loss_recons_joker', 'acc_guided', 'acc_joker',
                'joker_fraction', 'timing_contribution_ratio', 'train_objective',
            }
            self.assertTrue(expected <= keys, expected - keys)
            self.assertTrue(all(math.isfinite(value) for row in logged for value in row.values()))
            self.assertTrue(all(not row['encoder_training'] and not row['style_training']
                                for row in train_records))
            self.assertTrue(all(row['has_timing'] for row in eval_records))
            margins = [row['loss_margin'] for row in logged if 'loss_margin' in row]
            objectives = [row['train_objective'] for row in logged if 'train_objective' in row]
            for i, record in enumerate(train_records):
                self.assertAlmostEqual(margins[i], cfg['loss_margin'] * record['loss_margin'])
                self.assertAlmostEqual(objectives[i], record['loss_recons'])
            validation_losses = [row['val_loss'] for row in logged if 'val_loss' in row]
            for value, record in zip(validation_losses, eval_records[::2]):
                self.assertAlmostEqual(value, record['val_loss'])
            wandb_run.finish.assert_called_once()


class LiveIntegrationTest(unittest.TestCase):
    def test_midi_clock_uses_event_deltas_across_releases(self):
        source = (ROOT / 'interaction_buttons_style.py').read_text()
        callback = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'midiin_callback')
        clock = [100.0]
        received = []
        env = dict(time=SimpleNamespace(perf_counter=lambda: clock[0]),
                   midi_event_time=None, NOTE_ON=0x90, NOTE_OFF=0x80,
                   live_timing=object(),
                   buffer_lock=Lock(),
                   manageNote=lambda note, velocity, event_time: received.append(
                       (note, velocity, event_time)))
        exec(compile(ast.Module(body=[callback], type_ignores=[]), '<midi callback>', 'exec'), env)

        env['midiin_callback'](([0x90, 60, 90], 0.0))
        clock[0] += 1.0  # The previous note's inference occupied the callback.
        env['midiin_callback'](([0x80, 60, 0], 0.03))
        env['midiin_callback'](([0x90, 62, 90], 0.02))
        self.assertAlmostEqual(received[2][2] - received[0][2], 0.05)
        self.assertEqual([row[1] for row in received], [90, 0, 90])

    def test_toggle_pause_reset_recording_and_held_note_release(self):
        # Execute the real callbacks without opening MIDI/audio/GUI devices.
        source = (ROOT / 'interaction_buttons_style.py').read_text()
        names = {'manageNote', 'reset_context', '_on_key_press', '_on_key_release', 'make_controls_legend'}
        definitions = [node for node in ast.parse(source).body
                       if isinstance(node, ast.FunctionDef) and node.name in names]
        clock = [1.0]
        visualizer = Mock()
        detector = Mock()
        detector.update.return_value = False
        model = Mock()
        model.gen_pitch_token.side_effect = [(70, None), (72, None), (74, None)]
        env = dict(torch=torch, List=List, Tuple=Tuple, time=SimpleNamespace(perf_counter=lambda: clock[0]),
                   TIME_UNIT=TIME_UNIT, TRACES=False, USE_CACHE=False, CTX_LEN=4, TEMPERATURE=1,
                   TOTAL_GEN_LEN=4, device='cpu', model=model, visualizer=visualizer,
                   playNote=Mock(), to_device=lambda context, device: context,
                   key_to_button=lambda key: 2, JOKER_FORCED_KEYS=set(), joker_detector=detector,
                   timeLast=0, first_note=True, i=0, context=None, kv_cache=None, last_gen_time=0,
                   noteOn_dict={}, active_style_idx=1, style_contexts=[None, 'chosen style'],
                   style_context_masks_list=[None, None], cfg={'timing_enabled': True},
                   motif_style_ready=False, MOTIF_STYLE_KEY='6', buffer_lock=Lock(),
                   pkeyboard=SimpleNamespace(Key=SimpleNamespace(space='space')))
        for key in ('space_joker_held', 'joker_toggle_active', 'toggle_key_held', 'timing_key_held', 'memory_key_held'):
            env[key] = Event()
        env['dict_input_tokens'] = {'pitch': [60, 62, 64, 65], 'dtime': [0, 8, 8, 8], 'dur': [4] * 4}
        env['dict_output_tokens'] = {k: v + [0] * 4 for k, v in env['dict_input_tokens'].items()}
        env['b'] = [0, 1, 2, 3] + [0] * 4
        env['live_timing'] = LiveTimingContext([60, 62, 64, 65], [0, 1, 2, 3], [0, 8, 8, 8], 4)
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<live callbacks>', 'exec'), env)
        env['manageNote'](60, 90)
        env['kv_cache'] = object()
        env['_on_key_press'](SimpleNamespace(char='d'))
        self.assertIsNone(env['kv_cache'])
        self.assertFalse(env['live_timing'].enabled)
        clock[0] += .25
        env['manageNote'](61, 90)
        self.assertFalse(model.gen_pitch_token.call_args.args[0]['timing_mask'].any())
        clock[0] += 10
        env['manageNote'](62, 90)
        self.assertEqual(env['dict_output_tokens']['pitch'][4:7], [70, 72, 74])
        self.assertGreater(env['dict_output_tokens']['dtime'][6], 127)
        self.assertEqual(model.gen_pitch_token.call_args.args[0]['pitch'][0, :-1].tolist(), [60, 62, 64, 65])
        env['reset_context']()
        self.assertEqual(env['i'], 3)
        self.assertEqual(env['active_style_idx'], 1)
        self.assertFalse(env['live_timing'].enabled)
        env['manageNote'](60, 0)
        env['playNote'].assert_called_with(70, 0)
        self.assertNotIn(60, env['noteOn_dict'])


if __name__ == '__main__':
    unittest.main()
