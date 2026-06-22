use std::collections::HashMap;

/// A single reasoning node emitted by the local model.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: String,
    pub node_type: String,
    pub text: String,
    pub parent_id: Option<String>,
    pub confidence: f64,
}

/// In-memory reasoning graph. Each generation turn builds one graph; it is discarded
/// at turn end (no persistence needed for the demo).
pub struct ReasoningGraph {
    nodes: HashMap<String, Node>,
    insertion_order: Vec<String>, // preserves output sequence for display
    next_id: usize,
}

impl ReasoningGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            insertion_order: Vec::new(),
            next_id: 0,
        }
    }

    pub fn add_node(
        &mut self,
        node_type: &str,
        text: &str,
        parent_id: Option<String>,
        confidence: f64,
    ) -> String {
        let id = format!("n{}", self.next_id);
        self.next_id += 1;
        self.insertion_order.push(id.clone());
        self.nodes.insert(
            id.clone(),
            Node {
                id: id.clone(),
                node_type: node_type.to_string(),
                text: text.to_string(),
                parent_id,
                confidence,
            },
        );
        id
    }

    pub fn get_node(&self, id: &str) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// Walk from `id` up to the root, return the chain in root-first order.
    pub fn ancestor_chain(&self, id: &str) -> Vec<Node> {
        let mut chain = Vec::new();
        let mut current = id.to_string();
        loop {
            match self.nodes.get(&current) {
                Some(node) => {
                    chain.push(node.clone());
                    match &node.parent_id {
                        Some(p) => current = p.clone(),
                        None => break,
                    }
                }
                None => break,
            }
        }
        chain.reverse();
        chain
    }

    /// Replace a node's text and confidence after an advisor patch.
    pub fn patch_node(&mut self, id: &str, new_text: &str, new_confidence: f64) {
        if let Some(node) = self.nodes.get_mut(id) {
            node.text = new_text.to_string();
            node.confidence = new_confidence;
        }
    }

    /// All nodes in insertion order.
    pub fn all_nodes(&self) -> Vec<Node> {
        self.insertion_order
            .iter()
            .filter_map(|id| self.nodes.get(id).cloned())
            .collect()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ancestor_chain_order() {
        let mut g = ReasoningGraph::new();
        let a = g.add_node("claim", "A", None, 0.9);
        let b = g.add_node("claim", "B", Some(a.clone()), 0.8);
        let c = g.add_node("claim", "C", Some(b.clone()), 0.7);

        let chain = g.ancestor_chain(&c);
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[0].id, a);
        assert_eq!(chain[1].id, b);
        assert_eq!(chain[2].id, c);
    }

    #[test]
    fn test_patch_updates_text_and_confidence() {
        let mut g = ReasoningGraph::new();
        let id = g.add_node("claim", "original", None, 0.5);
        g.patch_node(&id, "patched", 0.95);
        let node = g.get_node(&id).unwrap();
        assert_eq!(node.text, "patched");
        assert_eq!(node.confidence, 0.95);
    }
}
