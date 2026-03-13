//! Simple frame profiler using browser Performance API
//!
//! Usage:
//!   profiler.begin("physics");
//!   // ... physics code ...
//!   profiler.end("physics");
//!
//!   profiler.report_if_needed(); // Logs every N frames

use std::collections::HashMap;
use web_sys::Performance;

/// Timing stats for a single section
#[derive(Default, Clone)]
struct SectionStats {
    total_ms: f64,
    count: u32,
    min_ms: f64,
    max_ms: f64,
    current_start: f64,
}

impl SectionStats {
    fn new() -> Self {
        SectionStats {
            min_ms: f64::MAX,
            max_ms: 0.0,
            ..Default::default()
        }
    }

    fn avg_ms(&self) -> f64 {
        if self.count > 0 {
            self.total_ms / self.count as f64
        } else {
            0.0
        }
    }
}

/// Frame profiler with named sections
pub struct Profiler {
    perf: Performance,
    sections: HashMap<&'static str, SectionStats>,
    frame_count: u32,
    report_interval: u32,
    enabled: bool,
}

impl Profiler {
    pub fn new(report_interval: u32) -> Self {
        let perf = web_sys::window()
            .expect("window")
            .performance()
            .expect("performance");

        Profiler {
            perf,
            sections: HashMap::new(),
            frame_count: 0,
            report_interval,
            enabled: true,
        }
    }

    /// Start timing a section
    #[inline]
    pub fn begin(&mut self, name: &'static str) {
        if !self.enabled { return; }
        let now = self.perf.now();
        self.sections.entry(name).or_insert_with(SectionStats::new).current_start = now;
    }

    /// End timing a section
    #[inline]
    pub fn end(&mut self, name: &'static str) {
        if !self.enabled { return; }
        let now = self.perf.now();
        if let Some(stats) = self.sections.get_mut(name) {
            let elapsed = now - stats.current_start;
            stats.total_ms += elapsed;
            stats.count += 1;
            stats.min_ms = stats.min_ms.min(elapsed);
            stats.max_ms = stats.max_ms.max(elapsed);
        }
    }

    /// Mark end of frame, report if interval reached
    pub fn end_frame(&mut self) {
        self.frame_count += 1;
        if self.frame_count >= self.report_interval {
            self.report();
            self.reset();
        }
    }

    /// Log profiling report to console
    fn report(&self) {
        use web_sys::console;

        let mut report = String::from("\n=== PROFILE (last 60 frames) ===\n");

        // Sort by total time descending
        let mut sections: Vec<_> = self.sections.iter().collect();
        sections.sort_by(|a, b| b.1.total_ms.partial_cmp(&a.1.total_ms).unwrap());

        for (name, stats) in sections {
            report.push_str(&format!(
                "{:20} avg:{:6.2}ms  min:{:6.2}ms  max:{:6.2}ms  total:{:7.1}ms\n",
                name,
                stats.avg_ms(),
                stats.min_ms,
                stats.max_ms,
                stats.total_ms
            ));
        }

        // Calculate total frame time
        let total: f64 = self.sections.values().map(|s| s.total_ms).sum();
        let avg_frame = total / self.frame_count as f64;
        let theoretical_fps = 1000.0 / avg_frame;

        report.push_str(&format!(
            "---\nTotal: {:.1}ms/frame → {:.0} FPS theoretical\n",
            avg_frame, theoretical_fps
        ));

        console::log_1(&report.into());
    }

    fn reset(&mut self) {
        self.sections.clear();
        self.frame_count = 0;
    }

    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
        let state = if self.enabled { "enabled" } else { "disabled" };
        web_sys::console::log_1(&format!("Profiler {}", state).into());
    }
}
