Dataset **CelebAMask-HQ** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/supervisely-supervisely-assets-public/teams_storage/N/i/Oz/bMyUfiVHflxjMS0Hfkf1bIOgrSMkM5wBKKrCh7zFWgp6Ep73JVQMKtik0RTKvLp5BuOWHz7fTAGT4fZvfQzPeKpMPzTJfhdAlAcR3wx4Jpox3ZwRfdo3YcjC3kkJ.tar)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='CelebAMask-HQ', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

