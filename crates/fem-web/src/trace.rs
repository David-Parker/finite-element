//! Tracing functionality for debugging simulation issues

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{console, Blob, Url};

/// Trace entry for debugging
#[derive(Clone, Debug)]
pub struct TraceEntry {
    pub frame: u32,
    pub ke: f32,
    pub min_j: f32,
    pub max_j: f32,
    pub max_vel: f32,
    pub max_force: f32,
    pub min_plastic_det: f32,
    pub max_plastic_det: f32,
}

/// Tracer for collecting simulation diagnostics
pub struct Tracer {
    pub enabled: bool,
    pub log: Vec<TraceEntry>,
}

impl Tracer {
    pub fn new() -> Self {
        Tracer {
            enabled: false,
            log: Vec::new(),
        }
    }

    pub fn toggle(&mut self) -> bool {
        self.enabled = !self.enabled;

        if self.enabled {
            self.log.clear();
            console::log_1(&"Tracing started (press T to stop and download)".into());
        } else {
            console::log_1(&format!("Tracing stopped. {} frames captured.", self.log.len()).into());
            self.download();
        }

        self.enabled
    }

    pub fn record(
        &mut self,
        frame: u32,
        ke: f32,
        min_j: f32,
        max_j: f32,
        max_vel: f32,
        max_force: f32,
        min_plastic_det: f32,
        max_plastic_det: f32,
    ) {
        if !self.enabled {
            return;
        }

        self.log.push(TraceEntry {
            frame,
            ke,
            min_j,
            max_j,
            max_vel,
            max_force,
            min_plastic_det,
            max_plastic_det,
        });
    }

    pub fn download(&self) {
        if self.log.is_empty() {
            console::log_1(&"No trace data to download".into());
            return;
        }

        // Build CSV content
        let mut csv = String::from("frame,ke,min_j,max_j,max_vel,max_force,min_plastic_det,max_plastic_det\n");
        for entry in &self.log {
            csv.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                entry.frame,
                entry.ke,
                entry.min_j,
                entry.max_j,
                entry.max_vel,
                entry.max_force,
                entry.min_plastic_det,
                entry.max_plastic_det,
            ));
        }

        // Create blob and download
        let array = js_sys::Array::new();
        array.push(&JsValue::from_str(&csv));

        if let Ok(blob) = Blob::new_with_str_sequence(&array) {
            if let Ok(url) = Url::create_object_url_with_blob(&blob) {
                if let Some(window) = web_sys::window() {
                    if let Some(document) = window.document() {
                        if let Ok(a) = document.create_element("a") {
                            let _ = a.set_attribute("href", &url);
                            let _ = a.set_attribute("download", "fem_trace.csv");
                            if let Ok(a) = a.dyn_into::<web_sys::HtmlElement>() {
                                a.click();
                            }
                            let _ = Url::revoke_object_url(&url);
                            console::log_1(&"Trace downloaded as fem_trace.csv".into());
                            return;
                        }
                    }
                }
            }
        }

        console::log_1(&"Failed to download trace".into());
    }
}

impl Default for Tracer {
    fn default() -> Self {
        Self::new()
    }
}
