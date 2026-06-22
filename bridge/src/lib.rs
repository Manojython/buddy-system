use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

mod entropy;
mod graph;
mod router;

use entropy::EntropyMonitor;
use graph::ReasoningGraph;
use router::Router;

// ---------------------------------------------------------------------------
// Helper: serialize a graph Node into a Python dict
// ---------------------------------------------------------------------------

fn node_to_dict(py: Python<'_>, node: &graph::Node) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("id", &node.id)?;
    d.set_item("type", &node.node_type)?;
    d.set_item("text", &node.text)?;
    d.set_item("parent_id", node.parent_id.as_deref())?;
    d.set_item("confidence", node.confidence)?;
    Ok(d.unbind().into_any())
}

// ---------------------------------------------------------------------------
// EntropyMonitor
// ---------------------------------------------------------------------------

#[pyclass(name = "EntropyMonitor")]
struct PyEntropyMonitor {
    inner: EntropyMonitor,
}

#[pymethods]
impl PyEntropyMonitor {
    /// threshold: entropy above which a boundary triggers an interrupt.
    /// min_tokens: minimum tokens in a window before an interrupt can fire.
    #[new]
    fn new(threshold: f64, min_tokens: usize) -> Self {
        Self {
            inner: EntropyMonitor::new(threshold, min_tokens),
        }
    }

    /// Feed one token. Returns True when generation should pause.
    fn update(&mut self, entropy: f64, is_boundary: bool) -> bool {
        self.inner.update(entropy, is_boundary)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Utility: compute entropy from a log-probability vector (post-softmax + log).
    #[staticmethod]
    fn entropy_from_logprobs(logprobs: Vec<f32>) -> f64 {
        entropy::entropy_from_logprobs(&logprobs)
    }

    /// Utility: compute entropy from raw logits (unnormalized).
    #[staticmethod]
    fn entropy_from_logits(logits: Vec<f32>) -> f64 {
        entropy::entropy_from_logits(&logits)
    }
}

// ---------------------------------------------------------------------------
// ReasoningGraph
// ---------------------------------------------------------------------------

#[pyclass(name = "ReasoningGraph")]
struct PyReasoningGraph {
    inner: ReasoningGraph,
}

#[pymethods]
impl PyReasoningGraph {
    #[new]
    fn new() -> Self {
        Self {
            inner: ReasoningGraph::new(),
        }
    }

    /// Add a node and return its id string.
    fn add_node(
        &mut self,
        node_type: &str,
        text: &str,
        parent_id: Option<String>,
        confidence: f64,
    ) -> String {
        self.inner.add_node(node_type, text, parent_id, confidence)
    }

    /// Return a node dict, or Python None if id not found.
    fn get_node(&self, id: &str, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.inner.get_node(id) {
            Some(n) => node_to_dict(py, n),
            None => Ok(py.None()),
        }
    }

    /// Return the ancestor chain from root to `id` as a list of dicts.
    fn ancestor_chain(&self, id: &str, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let chain = self.inner.ancestor_chain(id);
        let list = PyList::empty(py);
        for node in &chain {
            list.append(node_to_dict(py, node)?)?;
        }
        Ok(list.unbind().into_any())
    }

    /// Splice an advisor patch into a node in-place.
    fn patch_node(&mut self, id: &str, new_text: &str, new_confidence: f64) {
        self.inner.patch_node(id, new_text, new_confidence);
    }

    /// Return all nodes in insertion order as a list of dicts.
    fn all_nodes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let nodes = self.inner.all_nodes();
        let list = PyList::empty(py);
        for node in &nodes {
            list.append(node_to_dict(py, node)?)?;
        }
        Ok(list.unbind().into_any())
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

#[pyclass(name = "Router")]
struct PyRouter {
    inner: Router,
}

#[pymethods]
impl PyRouter {
    /// confidence_bar: minimum classical-tool confidence to trust its output.
    #[new]
    fn new(confidence_bar: f64) -> Self {
        Self {
            inner: Router::new(confidence_bar),
        }
    }

    fn register(&mut self, node_type: &str, tool_name: &str) {
        self.inner.register(node_type, tool_name);
    }

    /// Returns "classical:<tool_name>" or "cloud".
    fn decide(&self, node_type: &str) -> String {
        match self.inner.route(node_type) {
            router::Decision::ClassicalTool(name) => format!("classical:{name}"),
            router::Decision::CloudAdvisor => "cloud".to_string(),
        }
    }

    fn clears_bar(&self, tool_confidence: f64) -> bool {
        self.inner.clears_bar(tool_confidence)
    }

    fn confidence_bar(&self) -> f64 {
        self.inner.confidence_bar()
    }

    fn registered_types(&self) -> Vec<String> {
        self.inner.registered_types()
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn _bridge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEntropyMonitor>()?;
    m.add_class::<PyReasoningGraph>()?;
    m.add_class::<PyRouter>()?;
    Ok(())
}
