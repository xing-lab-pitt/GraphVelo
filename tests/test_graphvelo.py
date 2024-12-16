## test GraphVelo
import dynamo as dyn
from graph_velocity import GraphVelo


def test_graphvelo():
    simulator = dyn.sim.BifurcationTwoGenes(dyn.sim.bifur2genes_params, tau=1)
    simulator.simulate([0, 40], n_cells=2000)
    adata = simulator.generate_anndata()
    dyn.tl.neighbors(adata, basis='raw', n_neighbors=30)
    gv = GraphVelo(adata, xkey='total', vkey='velocity_T', approx=False)
    assert gv.X == adata.layers['total']
    assert gv.V == adata.layers['velocity_T']
    gv.train()
    assert gv.T is not None
    V_p = gv.project_velocity(adata.layers['total'])