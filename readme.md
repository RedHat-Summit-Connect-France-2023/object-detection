# Object detection

## Deployment

```
oc apply -k ./manifests/operators/
# Wait ~ 10min for operator to deploy
oc apply -k ./manifests/instances/object-detection
```
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
