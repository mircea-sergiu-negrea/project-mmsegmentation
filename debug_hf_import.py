# Debug test: import huggingface_hub.hf_api under debugpy launcher
import traceback
try:
    import huggingface_hub
    print('huggingface_hub.__file__ =', getattr(huggingface_hub, '__file__', None))
    from huggingface_hub import hf_api
    print('Imported hf_api as module:', hf_api)
    from huggingface_hub.hf_api import HfApi
    print('Imported HfApi:', HfApi)
except Exception as e:
    print('Exception during import:')
    traceback.print_exc()

input('done')
