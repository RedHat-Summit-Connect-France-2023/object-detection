# Object detection

## Deployment

```
oc apply -k ./manifests/operators/
# Wait ~ 10min for operator to deploy
oc apply -k ./manifests/instances/object-detection
```

Note: Sometimes model fail loading as the bucket is not ready yet. If you a connection refused error to minio port 9000, then restart the modelmesh pod to fix it.

## Labels

WIP on clusters and names. For now:

- tshirt, chemise
- manteau
- pull
- pantalon, bas
- chaussures
- ensemble
- lunettes
- cravatte
- montre
- foulard
- redhat
- sac
- parapluie
- accessoire
