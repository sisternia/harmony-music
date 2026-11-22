declare module "verovio/esm" {
  export type VerovioTimemapEntry = {
    tstamp: number;
    qstamp: number;
    tempo?: number;
    measureOn?: string;
    on?: string[];
    off?: string[];
    restsOn?: string[];
    restsOff?: string[];
  };

  export class VerovioToolkit {
    constructor(module: unknown);
    setOptions(options: Record<string, unknown>): void;
    loadData(data: string): boolean;
    getLog(): string;
    getPageCount(): number;
    renderToSVG(page: number): string;
    renderToTimemap(options?: Record<string, unknown>): VerovioTimemapEntry[];
  }
}

declare module "verovio/wasm" {
  const createModule: () => Promise<unknown>;
  export default createModule;
}
