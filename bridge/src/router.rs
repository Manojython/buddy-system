use std::collections::HashMap;

/// What the router decides to do with a flagged node.
#[derive(Debug, PartialEq)]
pub enum Decision {
    ClassicalTool(String), // tool name, e.g. "classifier", "ner"
    CloudAdvisor,
}

/// Maps node types to the name of a classical tool that can handle them.
/// Classical tool names are resolved by the Python layer (it owns the actual callables).
pub struct Router {
    registry: HashMap<String, String>, // node_type → tool_name
    confidence_bar: f64,               // minimum classical-tool confidence to trust its output
}

impl Router {
    pub fn new(confidence_bar: f64) -> Self {
        let mut registry = HashMap::new();
        // Seed the default registry — Python layer can override via register().
        registry.insert("classification".to_string(), "classifier".to_string());
        registry.insert("extraction".to_string(), "ner".to_string());
        registry.insert("similarity".to_string(), "similarity".to_string());
        registry.insert("sentiment".to_string(), "sentiment".to_string());
        // "claim", "sub_decision", "assumption" have no classical tool → fall to cloud.
        Self {
            registry,
            confidence_bar,
        }
    }

    pub fn register(&mut self, node_type: &str, tool_name: &str) {
        self.registry
            .insert(node_type.to_string(), tool_name.to_string());
    }

    /// First routing pass: returns which tool (if any) is registered for this node type.
    pub fn route(&self, node_type: &str) -> Decision {
        match self.registry.get(node_type) {
            Some(tool) => Decision::ClassicalTool(tool.clone()),
            None => Decision::CloudAdvisor,
        }
    }

    /// Second pass: did the classical tool's output clear the confidence bar?
    /// If not, the Python pipeline falls through to the cloud advisor.
    pub fn clears_bar(&self, tool_confidence: f64) -> bool {
        tool_confidence >= self.confidence_bar
    }

    pub fn confidence_bar(&self) -> f64 {
        self.confidence_bar
    }

    pub fn registered_types(&self) -> Vec<String> {
        self.registry.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_type_routes_to_classical() {
        let r = Router::new(0.8);
        assert_eq!(
            r.route("classification"),
            Decision::ClassicalTool("classifier".to_string())
        );
    }

    #[test]
    fn test_unknown_type_routes_to_cloud() {
        let r = Router::new(0.8);
        assert_eq!(r.route("sub_decision"), Decision::CloudAdvisor);
    }

    #[test]
    fn test_confidence_bar() {
        let r = Router::new(0.8);
        assert!(r.clears_bar(0.85));
        assert!(!r.clears_bar(0.75));
    }

    #[test]
    fn test_custom_registration() {
        let mut r = Router::new(0.7);
        r.register("my_type", "my_tool");
        assert_eq!(
            r.route("my_type"),
            Decision::ClassicalTool("my_tool".to_string())
        );
    }
}
