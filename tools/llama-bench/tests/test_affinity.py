import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pathlib import Path
import tempfile,unittest
from bench_affinity import snapshot,select,cpu_list

class AffinityPolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.sys=self.root/'sys';self.proc=self.root/'proc'
        self.write(self.sys/'devices/system/cpu/online','0-11')
        for cpu in range(12):
            p=self.sys/f'devices/system/cpu/cpu{cpu}/topology'
            self.write(p/'physical_package_id','0');self.write(p/'core_id',str(cpu%6))
            self.write(p/'thread_siblings_list',f'{cpu%6},{cpu%6+6}')
        self.write(self.proc/'interrupts','CPU0 CPU1\n127: 0 100 PCI-MSI kl1_dev0\n')
        self.write(self.proc/'irq/127/effective_affinity_list','11')
        self.write(self.proc/'irq/127/smp_affinity_list','0-11')
    def tearDown(self):self.temp.cleanup()
    def write(self,p,text):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
    def state(self,allowed=range(12)):return snapshot(self.sys,self.proc,allowed)
    def test_irq_sibling_outside_allowed(self):
        result=select('auto',self.state(range(6)))
        self.assertEqual(result['selected_cpu'],0)
        self.assertNotIn(5,result['safe_candidates']);self.assertEqual(result['excluded_irq_core_cpus'],[5,11])
    def test_allowed_and_physical_core(self):
        self.assertEqual(select('auto',self.state([2,5,8]))['safe_candidates'],[2])
        self.assertIsNone(select('auto',self.state([5,11]))['selected_cpu'])
    def test_explicit_zero_and_irq_warning(self):
        self.assertEqual(select('0',self.state())['selected_cpu'],0)
        self.assertEqual(select('2',self.state())['selected_cpu'],2)
        self.assertTrue(select('5',self.state())['warnings'])
    def test_invalid_explicit(self):
        for value in ['-1','auto2','1024','2,3','1.5']:
            with self.assertRaises(ValueError):select(value,self.state())
        with self.assertRaises(ValueError):select('6',self.state(range(6)))
    def test_unknown_irq_and_missing_effective(self):
        (self.proc/'irq/127/effective_affinity_list').unlink()
        self.assertIsNone(select('auto',self.state())['selected_cpu'])
        self.write(self.proc/'interrupts','CPU0 CPU1\n')
        self.assertIsNone(select('auto',self.state())['selected_cpu'])
        self.assertEqual(select('2',self.state())['selected_cpu'],2)
    def test_missing_topology(self):
        (self.sys/'devices/system/cpu/cpu11/topology/core_id').unlink()
        self.assertIsNone(select('auto',self.state())['selected_cpu'])
    def test_off_and_cpu_lists(self):
        self.assertIsNone(select('off',self.state())['selected_cpu'])
        self.assertEqual(cpu_list('0-2,8,11'),{0,1,2,8,11})
        for value in ['',',1','2-1','-1','1,','1-1048576']:
            with self.assertRaises(ValueError):cpu_list(value)

if __name__=='__main__':unittest.main()
