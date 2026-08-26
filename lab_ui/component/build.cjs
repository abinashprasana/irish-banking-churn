const fs = require("node:fs");
const path = require("node:path");
const zlib = require("node:zlib");

const projectRoot = path.resolve(__dirname, "..", "..");
const webModules = path.join(projectRoot, "web", "node_modules");
const ts = require(path.join(webModules, "typescript"));
const swc = require(path.join(webModules, "next", "dist", "build", "swc"));

const sourcePath = path.join(__dirname, "src", "decision-instrument.tsx");
const outputPath = path.join(
  __dirname,
  "..",
  "assets",
  "decision-instrument.bundle.js",
);

function packageFile(packageName, relativeFile) {
  const packageJson = require.resolve(`${packageName}/package.json`, {
    paths: [webModules],
  });
  return path.join(path.dirname(packageJson), relativeFile);
}

function moduleFactory(source) {
  return `function(module,exports,require){${source}\n}`;
}

async function main() {
  const source = fs.readFileSync(sourcePath, "utf8");
  const result = ts.transpileModule(source, {
    compilerOptions: {
      esModuleInterop: true,
      jsx: ts.JsxEmit.React,
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2020,
      sourceMap: false,
    },
    fileName: sourcePath,
    reportDiagnostics: true,
  });

  const errors = (result.diagnostics || []).filter(
    (diagnostic) => diagnostic.category === ts.DiagnosticCategory.Error,
  );
  if (errors.length) {
    for (const diagnostic of errors) {
      console.error(ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n"));
    }
    process.exitCode = 1;
    return;
  }

  const reactDomPackage = path.dirname(
    require.resolve("react-dom/package.json", { paths: [webModules] }),
  );
  const schedulerPackage = path.dirname(
    require.resolve("scheduler/package.json", { paths: [reactDomPackage] }),
  );
  const sources = {
    react: fs.readFileSync(packageFile("react", "cjs/react.production.js"), "utf8"),
    "react-dom": fs.readFileSync(
      packageFile("react-dom", "cjs/react-dom.production.js"),
      "utf8",
    ),
    "react-dom/client": fs.readFileSync(
      packageFile("react-dom", "cjs/react-dom-client.production.js"),
      "utf8",
    ),
    scheduler: fs.readFileSync(
      path.join(schedulerPackage, "cjs", "scheduler.production.js"),
      "utf8",
    ),
    entry: result.outputText,
  };

  const factories = Object.entries(sources)
    .map(([name, moduleSource]) => `${JSON.stringify(name)}:${moduleFactory(moduleSource)}`)
    .join(",");
  const unminified = `
const __modules={${factories}};
const __cache=Object.create(null);
function __require(id){
  if(__cache[id])return __cache[id].exports;
  const factory=__modules[id];
  if(!factory)throw new Error("Unknown component module: "+id);
  const module={exports:{}};
  __cache[id]=module;
  factory(module,module.exports,__require);
  return module.exports;
}
const __entry=__require("entry");
export default __entry.default;
`;

  await swc.loadBindings();
  const minified = await swc.minify(unminified, {
    compress: true,
    mangle: true,
    module: true,
  });
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  // The internal newline makes Streamlit classify this as inline source rather
  // than a file path; the bundle is still served directly from Python memory.
  fs.writeFileSync(
    outputPath,
    `/* Atlantic Ledger DecisionInstrument — generated; do not edit. */\n${minified.code}\n`,
    "utf8",
  );
  const rawBytes = Buffer.byteLength(minified.code);
  const gzipBytes = zlib.gzipSync(minified.code, { level: 9 }).byteLength;
  console.log(
    `Built ${path.basename(outputPath)} (${rawBytes} bytes, ${gzipBytes} bytes gzip)`,
  );
  if (gzipBytes > 120 * 1024) {
    throw new Error(`Component exceeds the 120 KB gzip budget: ${gzipBytes} bytes`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
