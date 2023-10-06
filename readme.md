# Object detection

## Deployment

**Modify the S3 secret** in ./manifests/instances/object-detection/inference-service/secret.yaml. **Change** AWS_S3_ENDPOINT with the minio route in the object-detection namespace.

### CPU

```
oc apply -k ./manifests/operators/base
# Wait ~ 10min for operator to deploy
oc apply -k ./manifests/instances/intelligent-application
oc apply -k ./manifests/instances/object-detection/inference-service
oc apply -k ./manifests/instances/object-detection/data-science-project/cpu
```

### GPU

```
oc apply -k ./manifests/operators/gpu
# Wait ~ 10min for operator to deploy
oc apply -k ./manifests/instances/intelligent-application
oc apply -k ./manifests/instances/object-detection/inference-service
oc apply -k ./manifests/instances/object-detection/data-science-project/gpu
```

Note: Sometimes model fail loading as the bucket is not ready yet. If you got a connection refused error to minio port 9000, then restart the modelmesh pod to fix it.

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