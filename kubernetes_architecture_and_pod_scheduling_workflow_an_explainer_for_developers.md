# Kubernetes Architecture and Pod Scheduling Workflow: An Explainer for Developers

## Core Kubernetes Architecture

![Diagram of core Kubernetes components and their interactions](images/k8s_core_architecture.png)
*Core Kubernetes architecture showing the API Server, etcd, Controller Manager, Scheduler, Kubelet, Kube‑proxy, and Container Runtime.*


The **API Server** is the unified entry point for all REST requests to the cluster. It authenticates users, applies admission controls, and routes requests to the appropriate backend components. Persistent cluster state is stored in **etcd**, a highly‑available, strongly consistent key‑value store that records every object definition, configuration, and status update. The **Controller Manager** runs a set of core controllers—such as the replication controller, endpoint controller, and node controller—that continuously reconcile the desired state defined in the API server with the actual state observed in the cluster. The **Scheduler** watches for Pods that lack an assigned node, evaluates them against a set of predicates (e.g., resource availability, taints) and priorities (e.g., affinity, load balancing), and selects the most suitable Node. On each Node, the **Kubelet** agent ensures that the containers running on the host match the PodSpec, monitors health, and reports status back to the API server. Finally, **Kube‑proxy** implements network service abstraction by configuring iptables or IPVS rules for Service traffic, while the **Container Runtime** (Docker, containerd, etc.) manages the lifecycle of the containers themselves.

## Pod Scheduling Workflow

![Step‑by‑step flowchart of the Kubernetes pod scheduling process](images/k8s_pod_scheduling_workflow.png)
*Pod scheduling workflow: filtering (predicates), scoring (priorities), optional extender layer, and binding to a node.*


Kubernetes scheduling starts with **filtering** (predicates). The scheduler examines every Node and discards those that break hard constraints: insufficient CPU or memory, mismatched `nodeSelector`, taints without matching tolerations, or violated pod‑affinity rules. Only Nodes that satisfy all predicates survive this phase.

Next comes **scoring** (priorities). The remaining Nodes are ranked by scoring functions such as `LeastAllocated` (prefers Nodes with the most free resources), `BalancedResourceAllocation` (spreads CPU and memory evenly), or any custom extenders. The Node with the highest aggregate score is selected as the best fit.

An **extender & policy** layer can augment both phases. Through the Scheduler Extender API, external services inject additional predicates or scores, allowing organizations to enforce bespoke placement policies without modifying the core scheduler.

Once a Node is chosen, the scheduler performs **binding**. It creates a `Binding` object that references the Pod and the target Node; the API server forwards this object to the Node’s kubelet, which then creates the Pod’s containers.

Below is a minimal custom scheduler configuration that **enables** the `NodeResourcesFit` predicate and **disables** the default pod‑affinity predicate:

```yaml
apiVersion: kubescheduler.config.k8s.io/v1
kind: KubeSchedulerConfiguration
profiles:
- schedulerName: custom-scheduler
  plugins:
    filter:
      enabled:
      - name: NodeResourcesFit
      disabled:
      - name: PodAffinity
```

Deploy this config via the `kube-scheduler` command line (`--config=/path/to/config.yaml`) to apply the custom behavior.

## Scheduling Edge Cases & Failure Modes

- **Resource Exhaustion** – Pods stay pending when every node lacks sufficient CPU or memory for the request. Run `kubectl top nodes` to spot nodes with zero allocatable resources and consider scaling the cluster or adjusting resource requests.  
- **Node Selectors / Affinity Mismatches** – If a pod’s `nodeSelector` or affinity rules reference labels that no node currently has, the scheduler cannot place the pod. Verify the target labels with `kubectl get nodes --show-labels`.  
- **Taints & Tolerations** – A node marked with a `NoSchedule` (or similar) taint will reject pods lacking a matching toleration. Use `kubectl describe node <name>` to view active taints and compare them against the pod’s tolerations.  
- **TopologySpreadConstraints** – When spread constraints require pods to be evenly distributed across zones or racks, and the topology cannot satisfy the rule, the pod remains unscheduled. Check the constraint definition and the current pod distribution.  
- **Scheduler Extender Errors** – External scheduler extenders may return errors or timeouts, causing the pod to be flagged unschedulable. Review extender logs and the scheduler’s event stream for extender‑related messages.  
- **API Server / etcd Latency** – High latency or stale reads from the API server or etcd can produce transient scheduling failures. Monitor API server latency metrics and etcd health to detect such conditions.

## Observability & Debugging Pod Scheduling

- `kubectl describe pod <name>` – Shows the pod’s event timeline; look for “FailedScheduling” messages that list node predicates or taints that caused the scheduler to reject each candidate node.  
- Scheduler Logs – Deploy the scheduler with `--v=4` (or higher) to emit detailed predicate and priority evaluation logs; these reveal why a pod was filtered out or deprioritized.  
- Metrics Server & `kubectl top pods` – Verify that the pod’s CPU and memory requests fit within the cluster’s allocatable resources; mismatched requests often leave pods pending.  
- Cluster Autoscaler – Check if pending pods trigger node provisioning; inspect the autoscaler logs for “scale‑up” events or errors that prevented new nodes from being added.  
- Profiling & Tracing – Access the scheduler’s `/debug/pprof` endpoint (e.g., `curl http://<scheduler-ip>:10251/debug/pprof/profile`) to capture CPU or heap profiles and identify performance bottlenecks in scheduling cycles.  
- Etcd Health – Run `etcdctl endpoint health` (or `etcdctl endpoint status`) to confirm that the key‑value store is responsive; an unhealthy etcd can delay state propagation and stall scheduling decisions.
