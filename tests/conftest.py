def pytest_sessionstart(session):
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(1)
