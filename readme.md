# Object detection

## Deployment

**Modify the S3 secret**. Go to ./manifests/instances/object-detection/inference-service/secret.yaml. **Change** AWS_S3_ENDPOINT. **Change or delete** if you don't use AWS_S3_INTERNAL_ENDPOINT

```
oc apply -k ./manifests/operators/
# Wait ~ 10min for operator to deploy
oc apply -k ./manifests/instances/object-detection
```

Note: Sometimes model fail loading as the bucket is not ready yet. If you a connection refused error to minio port 9000, then restart the modelmesh pod to fix it.

## Labels

WIP on clusters and names. For now:

- haut à manches courtes
- haut à manches longues
- veste à manches courtes
- veste à manches longues
- gillet
- écharpe
- short
- pantalon
- jupe
- robe à manche courtes
- robe à manches longues
- robe