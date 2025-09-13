import { JSDOM } from 'jsdom';
import { readFileSync } from 'fs';

// Set up DOM environment for Node.js
const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>');
global.window = dom.window;
global.document = dom.window.document;
global.DOMParser = dom.window.DOMParser;
global.Node = dom.window.Node;

// DOM fingerprint function
function calculateDOMChangePercentage(dom1, dom2) {
  const parser = new DOMParser();
  const doc1 = typeof dom1 === "string" ? parser.parseFromString(dom1, "text/html") : dom1;
  const doc2 = typeof dom2 === "string" ? parser.parseFromString(dom2, "text/html") : dom2;

  const IGNORED_TAGS = new Set(["script", "style", "meta", "link", "title", "noscript"]);
  const ALLOWED_ATTRS = new Set(["role", "type", "aria-label", "aria-hidden", "id", "class"]);

  function cleanTree(root) {
    function traverse(node) {
      if (!node) return null;
      
      if (node.nodeType === Node.TEXT_NODE || node.nodeType === Node.COMMENT_NODE) {
        return null;
      }
      if (node.nodeType !== Node.ELEMENT_NODE) return null;

      const tag = node.tagName.toLowerCase();
      if (IGNORED_TAGS.has(tag)) return null;

      let attrs = {};
      if (node.attributes) {
        for (let attr of node.attributes) {
          if (ALLOWED_ATTRS.has(attr.name)) {
            attrs[attr.name] = attr.value;
          }
        }
      }

      const children = [];
      if (node.childNodes) {
        for (let child of node.childNodes) {
          const childTree = traverse(child);
          if (childTree) children.push(childTree);
        }
      }

      return { tag, attrs, children };
    }
    
    return traverse(root);
  }

  const body1 = doc1.body || doc1.documentElement;
  const body2 = doc2.body || doc2.documentElement;
  
  const tree1 = cleanTree(body1);
  const tree2 = cleanTree(body2);

  function extractFeatures(tree) {
    const tagCounts = {};
    const edges = new Set();
    const layoutTags = new Set(["header", "footer", "nav", "main", "section", "article"]);
    const layoutPresence = new Set();
    const attrs = {};

    function dfs(node, parentTag) {
      if (!node) return;
      
      tagCounts[node.tag] = (tagCounts[node.tag] || 0) + 1;

      if (parentTag) {
        edges.add(`${parentTag}->${node.tag}`);
      }

      if (layoutTags.has(node.tag)) {
        layoutPresence.add(node.tag);
      }

      for (let [k, v] of Object.entries(node.attrs)) {
        const key = `${node.tag}:${k}=${v}`;
        attrs[key] = (attrs[key] || 0) + 1;
      }

      for (let child of node.children) {
        dfs(child, node.tag);
      }
    }
    
    dfs(tree, null);
    return { tagCounts, edges, layoutPresence, attrs };
  }

  const f1 = extractFeatures(tree1);
  const f2 = extractFeatures(tree2);

  function jaccard(a, b) {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
    let intersection = 0, union = 0;
    for (let k of keys) {
      const va = a[k] || 0;
      const vb = b[k] || 0;
      intersection += Math.min(va, vb);
      union += Math.max(va, vb);
    }
    return union === 0 ? 1 : intersection / union;
  }

  function jaccardSet(a, b) {
    const setA = new Set(a);
    const setB = new Set(b);
    const intersection = new Set([...setA].filter(x => setB.has(x)));
    const union = new Set([...setA, ...setB]);
    return union.size === 0 ? 1 : intersection.size / union.size;
  }
  console.log("f1.tagCounts->", JSON.stringify(f1.tagCounts));
  console.log("f2.tagCounts->", JSON.stringify(f2.tagCounts));
  console.log("f1.edges->", JSON.stringify([...f1.edges]));
  console.log("f2.edges->", JSON.stringify([...f2.edges]));
  console.log("f1.layoutPresence->", JSON.stringify([...f1.layoutPresence]));
  console.log("f2.layoutPresence->", JSON.stringify([...f2.layoutPresence]));
  console.log("f1.attrs (first 10)->", JSON.stringify(Object.entries(f1.attrs).slice(0, 10)));
  console.log("f2.attrs (first 10)->", JSON.stringify(Object.entries(f2.attrs).slice(0, 10)));

  const structureChange = 1 - jaccard(f1.tagCounts, f2.tagCounts);
  const hierarchyChange = 1 - jaccardSet(f1.edges, f2.edges);
  const layoutChange = 1 - jaccardSet(f1.layoutPresence, f2.layoutPresence);
  const attributeChange = 1 - jaccard(f1.attrs, f2.attrs);

  const overallChange =
    structureChange * 0.4 +
    hierarchyChange * 0.3 +
    layoutChange * 0.2 +
    attributeChange * 0.1;

  let decision;
  if (overallChange <= 0.15) decision = "SAFE";
  else if (overallChange <= 0.35) decision = "CAUTION";
  else if (overallChange <= 0.50) decision = "RISKY";
  else decision = "DANGEROUS";

  return {
    change: +(overallChange * 100).toFixed(2),
    decision,
    metrics: {
      structureChange: +(structureChange * 100).toFixed(2),
      hierarchyChange: +(hierarchyChange * 100).toFixed(2),
      layoutChange: +(layoutChange * 100).toFixed(2),
      attributeChange: +(attributeChange * 100).toFixed(2),
    },
  };
}

// Get file paths from command line
const args = process.argv.slice(2);

if (args.length !== 2) {
  console.log('Usage: node compare-real-doms.js <expectedFile> <actualFile>');
  process.exit(1);
}

try {
  const expectedDOM = readFileSync(args[0], 'utf8');
  const actualDOM = readFileSync(args[1], 'utf8');
  
  const result = calculateDOMChangePercentage(expectedDOM, actualDOM);
  
  console.log(`Change: ${result.change}%`);
  console.log(`Decision: ${result.decision}`);
  console.log(`Structure: ${result.metrics.structureChange}%`);
  console.log(`Hierarchy: ${result.metrics.hierarchyChange}%`);
  console.log(`Layout: ${result.metrics.layoutChange}%`);
  console.log(`Attributes: ${result.metrics.attributeChange}%`);
  
} catch (error) {
  console.error('Error:', error.message);
  process.exit(1);
}