# Kubernetes Architecture and Pod Scheduling Workflow

---

## Core Kubernetes Architecture

Kubernetes is built around a **control‑plane** that manages the desired state of the cluster and a set of **node components** that run the workloads. The control‑plane components communicate via the API server and store cluster state in etcd.

![Kubernetes control plane and node architecture diagram](images/k8s_architecture.png)
*Figure 1: Core Kubernetes architecture showing control‑plane and node components.*

- **API Server** – Central REST endpoint for all cluster interactions.
- **etcd** – Consistent, highly‑available key‑value store for configuration and state.
- **Controller Manager** – Runs controllers that reconcile the actual state with the desired state (e.g., Deployment controller, Node controller).
- **Scheduler** – Assigns newly created Pods to suitable Nodes based on constraints and policies.
- **Node Components** (run on every worker node)
  - **kubelet** – Agent that ensures containers are running as defined in PodSpecs.
  - **kube‑proxy** – Handles network routing for Services.
  - **Container Runtime** – Executes containers (Docker, containerd, etc.).

---

## Pod Scheduling Workflow

When a Pod is created, the Scheduler follows a multi‑step process to decide where it should run. The steps are:

1. **Watch for Unscheduled Pods** – The Scheduler watches the API server for Pods without a `nodeName`.
2. **Filtering (Predicates)** – Eliminate Nodes that don’t satisfy hard constraints (e.g., insufficient CPU/memory, node selector, taints/tolerations).
3. **Scoring (Priorities)** – Rank the remaining Nodes using scoring functions (e.g., node affinity, pod affinity/anti‑affinity, resource balance).
4. **Select the Best Node** – Choose the highest‑scoring Node (ties broken randomly).
5. **Bind** – Write the selected Node name back to the Pod’s `spec.nodeName` field, effectively binding the Pod.
6. **Update** – The kubelet on the chosen Node sees the binding and starts creating the containers.

![Kubernetes pod scheduling workflow flowchart](images/pod_scheduling_workflow.png)
*Figure 2: Pod scheduling workflow performed by the Kubernetes Scheduler.*

---

## Hands‑On: Minimal Pod Spec and Observing Scheduling

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hello-world
spec:
  containers:
  - name: hello
    image: nginx:alpine
    resources:
      requests:
        cpu: "100m"
        memory: "128Mi"
```

```bash
# Apply the Pod
kubectl apply -f hello-world.yaml

# Watch where it lands
kubectl get pod hello-world -o wide
```

The `-o wide` output shows the `NODE` column, confirming which node the Scheduler chose.

---

## Debugging Scheduling Failures & Edge Cases

Common reasons a Pod stays **Pending**:

- **Insufficient resources** – No node has enough CPU/memory.
- **Node selector / affinity mismatch** – Constraints don’t match any node labels.
- **Taints & tolerations** – Pod lacks a toleration for a tainted node.
- **Pod anti‑affinity** – Conflicts with existing Pods on all nodes.
- **Quota limits** – Namespace or ResourceQuota blocks the request.

**Troubleshooting steps**:

```bash
kubectl describe pod <pod-name>
# Look for the "Events" section – it lists why the scheduler rejected each node.
```

You can also simulate scheduling decisions with:

```bash
kubectl dry-run=client -o yaml apply -f <pod.yaml>
```

---

## Performance & Cost Implications of Scheduling

- **Bin‑packing vs. spreading** – Tight packing maximizes resource utilization (lower cost) but can increase risk of simultaneous failures. Spreading improves resilience at the expense of higher resource waste.
- **Node sizing** – Larger nodes reduce per‑pod overhead but may lead to under‑utilization if workloads are small and bursty.
- **Spot/Preemptible nodes** – Scheduling non‑critical Pods onto cheaper, preemptible instances can cut costs, but you must handle possible evictions.
- **Custom scheduler policies** – Tailor scoring functions to prioritize cost‑effective nodes, GPU availability, or latency requirements.

Understanding the Scheduler’s decisions helps you balance cost, performance, and reliability in production clusters.
