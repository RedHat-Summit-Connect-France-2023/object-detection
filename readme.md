# Object detection

## Deployment

```
oc apply -k ./manifests/operators/
# Wait ~ 10min for operator to deploy
oc apply -k ./manifests/instances/object-detection
```
