# Demystifying Kubernetes Architecture and the Pod Scheduling Workflow

## Core Control‑Plane Components that Drive Scheduling

![Diagram of Kubernetes core control plane components and their interactions](images/k8s_control_plane_architecture.jpg)
*Figure 1: Kubernetes control‑plane architecture illustrating the API server, etcd, controller manager, scheduler, and kubelet.*


- **API Server** – the single entry point for all cluster requests, performing admission control, validating pod specifications, and storing them in etcd. It exposes the current state via its REST API and pushes updates to the Scheduler and Controller Manager.  
- **etcd** – a highly available, distributed key‑value store that persists every pod object, node status, and scheduler binding. It guarantees durability and consistency, serving as the definitive source of truth for the cluster.  
- **Controller Manager** – hosts a suite of controllers (e.g., replication, node, endpoint) that continually reconcile the desired state with the actual state, issuing updates back to the API Server and ensuring resources remain in sync.  
- **Scheduler** – watches the API Server for unscheduled pods, queries node metrics and topology information, evaluates scheduling predicates and priorities, and binds each pod to the most suitable node.  
- **Kubelet** – runs on every node, reporting capacity, health, and resource usage to the API Server, and enforcing the binding by launching the pod’s containers and maintaining their lifecycle.

## Step‑by‑Step Pod Scheduling Workflow

![Flowchart of the Kubernetes pod scheduling workflow](images/pod_scheduling_workflow.jpg)
*Figure 2: Step‑by‑step workflow of pod scheduling from pod creation to node activation.*


1. **Watch for Unschedulable Pods**  
   The kube‑scheduler subscribes to the API server and reacts to Pod creations whose `.spec.nodeName` field is empty. Each such Pod enters the scheduling pipeline.

2. **Filter Phase (Predicates)**  
   The scheduler evaluates every node against a set of predicates. Nodes lacking the required CPU, memory, or other resources are discarded. Taints that the Pod does not tolerate, and affinity/anti‑affinity constraints that cannot be satisfied, also cause removal. Only nodes that survive all predicates proceed to scoring.

3. **Score Phase (Priorities)**  
   Remaining nodes receive a numerical score based on priority functions. Common functions include *LeastAllocated* (favor lightly loaded nodes) and *BalancedResourceUsage* (evenly spread CPU/memory). The scheduler may combine multiple functions, each weighted appropriately.

4. **Select and Bind**  
   The node with the highest aggregate score is chosen. The scheduler creates a Binding object (`apiVersion: v1`, `kind: Binding`) that records the Pod’s assignment to the selected node and posts it to the API server.

5. **Node Activation**  
   The kubelet on the chosen node watches for new bindings. Upon receiving a binding, it pulls the Pod’s container images, sets up the runtime environment, and launches the containers.

6. **Minimal Pod Manifest (YAML)**  
   ```yaml
   apiVersion: v1
   kind: Pod
   metadata:
     name: demo-pod
   spec:
     containers:
     - name: nginx
       image: nginx:latest
       resources:
         requests:
           cpu: 100m
           memory: 200Mi
   ```
   The manifest includes the required fields: `apiVersion`, `kind`, `metadata.name`, and a container list with `name`, `image`, and optional `resources.requests`. Once applied, the scheduler will place this Pod according to the workflow above.

## Common Edge Cases and Failure Modes

1. **Insufficient CPU/Memory** – Pods remain pending when no node reports enough allocatable CPU or memory to satisfy their requests. The scheduler cannot bind the pod, and it stays in the Pending state until resources free up or a node with sufficient capacity is added.

2. **Taints/Tolerations mismatches** – A node’s taint blocks any pod that does not carry a matching toleration. Even if the node has free resources, the pod will not be scheduled, leading to persistent Pending status.

3. **Node Affinity conflicts** – Contradictory `requiredDuringSchedulingIgnoredDuringExecution` rules can eliminate all candidate nodes. If a pod demands both `zone=us-west` and `zone=us-east`, the scheduler finds no match and leaves the pod unscheduled.

4. **Pod Priority & Preemption** – High‑priority pods may evict lower‑priority ones to reclaim resources. This can trigger a cascade of evictions, causing multiple pods to repeatedly reschedule and potentially destabilize the workload.

5. **Scheduler crashes or misconfiguration** – If the scheduler pod crashes or the policy file is missing/incorrect, no scheduling decisions are made. Pods remain pending indefinitely until the scheduler recovers or the configuration is corrected.
