import config
import network2
import torch

import os

def load_model(model, continue_id):
    filename = f'{type(model).__name__}_{continue_id}.pth'
    state_dict = torch.load(os.path.join(config.MODELS_DIR, filename))
    model.load_state_dict(state_dict)
    return model

E = network2.Embedder()
G = network2.Generator()

example = torch.rand(1, 3, 256, 256)
example2 = torch.rand(1, 512)

E = load_model(E, "20191110_1511")
G = load_model(G, "20191110_1511")

E.eval()
G.eval()

with torch.jit.optimized_execution(True):
    traced_script_E = torch.jit.trace(E, (example,example))#,check_trace=True)
    traced_script_G = torch.jit.trace(G, (example,example2))#,check_trace=False)

#print(traced_script_E.code)
#print(traced_script_G.code)

traced_script_E.save("./assets/E_old.pt")
traced_script_G.save("./assets/G_old.pt")
