import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import kunlun_guarded_runner as runner

class GuardFailureTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.out=self.root/'results/test'
        guard=self.root/'install/bin/k200-thermal-guard';guard.parent.mkdir(parents=True);guard.write_text('not executed')
        checker=self.root/'checker.py';checker.write_text('not executed')
        runner.configure(self.root,self.out,check_users=checker)
    def tearDown(self):self.temp.cleanup()
    def test_unreadable_state_is_unhealthy(self):
        with patch.object(Path,'read_text',side_effect=OSError('simulated proc read failure')):
            self.assertFalse(runner.device_healthy())
            self.assertTrue(all(x.startswith('unavailable:') for x in runner.states().values()))
    def test_error_field_read_failure_is_unhealthy(self):
        with patch.object(runner,'states',return_value={0:'RUNNING',1:'RUNNING'}),patch.object(Path,'read_text',side_effect=OSError('simulated errinfo failure')):
            self.assertFalse(runner.device_healthy())
    def test_thermal_latch_refuses_launch(self):
        logs=self.out/'logs';logs.mkdir();(logs/'THERMAL_ABORT').write_text('test latch')
        with patch.object(runner.subprocess,'run') as launch:
            with self.assertRaises(RuntimeError):runner.run('latch-test',['never-execute'])
            launch.assert_not_called()
    def test_preflight_read_failure_latches_before_launch(self):
        with patch.object(Path,'read_text',side_effect=OSError('simulated device unavailable')),patch.object(runner.subprocess,'run') as launch:
            with self.assertRaises(RuntimeError):runner.run('missing-device-test',['never-execute'])
            launch.assert_not_called()
        self.assertTrue((self.out/'XPU_ABORT').is_file())
        self.assertIn('preflight_read_error',(self.out/'XPU_ABORT').read_text())

if __name__=='__main__':unittest.main()
