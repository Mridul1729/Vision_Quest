Place the existing trained portsync_model.pkl in this folder.

The API will automatically load it if present.
If it is absent, all non-ML endpoints still work, while /predict returns
a clear model-not-found message.

Do not replace the existing trained model with a fabricated model for the demo.
