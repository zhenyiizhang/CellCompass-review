"""CPU installation check: engine transport, a fitted toy field, and model loading."""
from pathlib import Path
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'CytoBridge-main')]

def main():
    started=time.perf_counter()
    import numpy as np
    import torch
    import yaml
    from CytoBridge.tl.flow_matching import OTPlanSampler, ConditionalFlowMatcher
    from CytoBridge.tl.training_algorithm import TrainingAlgorithmContext
    from CytoBridge.tl.training_algorithm_registry import list_training_algorithms,load_training_algorithm
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(1)
    x=torch.randn(32,2)
    y=x+torch.tensor([1.,-.5])
    sampler=OTPlanSampler('exact')
    plan=sampler.get_map(x,y)
    assert np.isfinite(plan).all() and (plan>=0).all()
    assert np.allclose(plan.sum(axis=0),1/32) and np.allclose(plan.sum(axis=1),1/32)
    matcher=ConditionalFlowMatcher(sigma=0.0)
    t,z,u=matcher.sample_location_and_conditional_flow(x,y)
    assert torch.isfinite(z).all() and torch.allclose(u,y-x)
    # A deliberately simple field makes the expected displacement explicit.
    velocity=torch.nn.Parameter(torch.zeros(2))
    optimizer=torch.optim.SGD([velocity],lr=.2)
    for _ in range(60):
        optimizer.zero_grad();loss=((velocity-u)**2).mean();loss.backward();optimizer.step()
    predicted=x+velocity.detach()
    error=float((predicted-y).abs().max())
    assert error<1e-4,error
    records=list_training_algorithms([ROOT/'algorithms'])
    assert len(records)==4,len(records)
    loaded=[]
    for record in records:
        item=load_training_algorithm(record.algorithm_id,[ROOT/'algorithms'])
        config=yaml.safe_load((item.root_dir/'config.yaml').read_text())
        context=TrainingAlgorithmContext(algorithm_id=item.algorithm_id,input_adata_path='',
            output_dir=str(ROOT/'output/installation_check'),stage='pilot',base_config_name=None,
            resolved_base_config=config,metadata={'algorithm_dir':str(item.root_dir)})
        spec=item.build_spec(context)
        assert spec.algorithm_id==record.algorithm_id
        loaded.append(record.algorithm_id)
    result={'status':'passed','device':'cpu','synthetic_cells':32,'max_endpoint_error':error,
            'algorithms_loaded':loaded,'elapsed_seconds':round(time.perf_counter()-started,2),
            'scope':'Installation check using synthetic data and model loading.'}
    out=ROOT/'output';out.mkdir(exist_ok=True)
    (out/'installation_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
