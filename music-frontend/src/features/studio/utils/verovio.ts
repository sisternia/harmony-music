export type Toolkit = import("verovio/esm").VerovioToolkit;

let toolkitPromise: Promise<Toolkit> | null = null;

export function loadVerovioToolkit(): Promise<Toolkit> {
  if (!toolkitPromise) {
    toolkitPromise = (async () => {
      const [{ default: createModule }, { VerovioToolkit }] = await Promise.all([
        import("verovio/wasm"),
        import("verovio/esm"),
      ]);
      return new VerovioToolkit(await createModule());
    })();
    toolkitPromise.catch(() => {
      toolkitPromise = null;
    });
  }
  return toolkitPromise;
}

