import importlib.util, math, unittest
from pathlib import Path
import proyecto_ag as p

class ProjectTests(unittest.TestCase):
    def test_exact_square(self):
        d=p.matrix([(0,0),(1,0),(1,1),(0,1)])
        optimum,route=p.exact(d)
        self.assertAlmostEqual(optimum,4.0)
        self.assertTrue(p.valid(route,3))
    def test_closed_route(self):
        d=p.matrix([(0,0),(3,0),(0,4)])
        self.assertAlmostEqual(p.distance([0,1],d),12.0)
    def test_bad_permutation(self):
        self.assertFalse(p.valid([0,0,2],3))
    def check_backend(self,backend):
        points=p.read_data(Path(__file__).with_name('clientes.csv'))
        d=p.matrix(points)
        a,h=p.run_ga(d,7,backend); b,j=p.run_ga(d,7,backend)
        self.assertEqual(a,b); self.assertEqual(h,j)
        self.assertEqual(len(h),31)
        self.assertTrue(p.valid(a,12))
        self.assertTrue(all(h[i+1]['best_km']<=h[i]['best_km']+1e-9 for i in range(30)))
    def test_reference(self): self.check_backend('reference')
    @unittest.skipUnless(importlib.util.find_spec('deap'),'DEAP no instalado; validación pendiente')
    def test_deap(self): self.check_backend('deap')

if __name__=='__main__': unittest.main(verbosity=2)
