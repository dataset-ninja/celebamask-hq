Dataset **CelebAMask-HQ** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/remote/eyJsaW5rIjogImZzOi8vYXNzZXRzLzIwNDFfQ2VsZWJBTWFzay1IUS9jZWxlYmFtYXNrLWhxLURhdGFzZXROaW5qYS50YXIiLCAic2lnIjogIlduVmIvQW1YYk5BdGtSUzVoK0EzTkxqMUpZSEFRbFphR3RYV3ZBMmFxKzQ9In0=)

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

