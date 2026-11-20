# Inside Kubernetes: Architecture and the Pod Scheduling Workflow

## Kubernetes Control‑Plane Architecture

The control plane is the brain of a cluster, orchestrating desired state and reconciling it with reality.

- **API Server** – The sole entry point for all RESTful requests. It validates and authenticates calls, serializes objects to the etcd key‑value store, and serves the cluster’s current state to clients and internal components.  
- **Scheduler** – Continuously watches for unscheduled pods. Its placement loop follows four stages:  
  1. **Watch** – Detects new pods.  
  2. **Filter** – Excludes nodes that fail resource, topology, or taint tolerations.  
  3. **Score** – Rates remaining nodes using predicates like least‑utilized or affinity.  
  4. **Bind** – Persists the chosen node assignment in etcd.  
- **Controller Manager** – Runs a set of controllers that reconcile actual cluster state with the desired state. Key defaults include:  
  - *Node controller* monitors node health and updates status.  
  - *Replication controller* (or ReplicaSet) ensures the correct number of pod replicas.  
  - *Endpoint controller* maintains service‑to‑pod mappings, which indirectly influence scheduling decisions.  
- **Cloud‑Controller Manager** – Optional, bridges the cluster with cloud provider APIs, provisioning load balancers, updating node IPs, or attaching persistent volumes.  
- **etcd** – The persistent datastore holding the desired state (objects created via the API server). Every control‑plane component reads from etcd to know the current cluster configuration and writes updates—such as node status, pod assignments, or controller reconciliations—to keep the cluster in sync.



## Extending the Scheduler with a Simple Custom Plugin

To keep pods from landing on nodes marked for maintenance, add a lightweight predicate to the kube‑scheduler.

1. **Create a Go module and import the framework**  
   ```bash
   mkdir maintenance-filter && cd maintenance-filter
   go mod init maintenance-filter
   go get k8s.io/kube-scheduler/framework@latest
   ```

2. **Implement the `FilterPlugin`**  
   ```go
   package main

   import (
       "context"
       "k8s.io/kubernetes/pkg/scheduler/framework"
       corev1 "k8s.io/api/core/v1"
   )

   type NodeMaintenanceFilter struct{}

   func (f *NodeMaintenanceFilter) Name() string { return "NodeMaintenanceFilter" }

   func (f *NodeMaintenanceFilter) Filter(ctx context.Context, state *framework.CycleState,
       pod *corev1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
       if val, ok := nodeInfo.Node().Labels["maintenance"]; ok && val == "true" {
           return framework.NewStatus(framework.Unschedulable, "node in maintenance")
       }
       return framework.NewStatus(framework.Success)
   }
   ```

3. **Register in `scheduler-config.yaml`**  
   ```yaml
   profiles:
   - schedulerName: default-scheduler
     plugins:
       filter:
         enabled:
         - name: NodeMaintenanceFilter
   ```

4. **Deploy the custom scheduler**  
   Bundle the binary in a side‑car Deployment, mount the config file, and start the scheduler with `--config=/etc/scheduler/scheduler-config.yaml`.

5. **Verify**  
   After scheduling a pod, `kubectl describe pod <name>` should show `Unschedulable: node in maintenance` in the events section, confirming the predicate is active.

## Observability, Debugging, and Edge Cases

Diagnosing pod scheduling failures starts with the Kubernetes event stream. Running `kubectl get events --field-selector reason=FailedScheduling` surfaces predicate rejections such as resource constraints, node affinity mismatches, or unsatisfied pod priority. For deeper insight, enable scheduler profiling with `--profiling` and hit the `/debug/pprof` endpoints. This exposes CPU hotspots in both the built‑in scheduler and any custom scheduling plugins you’ve integrated.

Performance telemetry is equally critical. The `scheduler_binding_latency_seconds` metric, exposed by the kube‑scheduler Prometheus endpoint, quantifies the time from pod admission to binding. Monitoring this metric alerts you when binding stalls or when custom predicates add unacceptable overhead. Complement this with the `scheduler_evaluation_duration_seconds` histogram to see how long the scheduler spends evaluating predicates per pod.

Edge cases often masquerade as simple failures. A pod that references a `nodeSelector` with no matching node, or requests more CPU/memory than any node offers, will repeatedly hit the “No nodes available” error. Taint and toleration misconfigurations also prevent binding. Before diving into logs, run `kubectl describe node` on the candidate nodes to verify resource availability and taint lists.

Finally, consider the cost side of scheduling. Over‑constraining pods—by using too many node selectors, strict affinity rules, or high resource limits—can lead to under‑utilized nodes. This waste translates directly into higher cloud spend. Balancing constraints with resource utilization is essential for both performance and economics.
