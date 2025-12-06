import { describe, it, expect } from "@jest/globals";
import { ensureConfiguration } from "../../configuration.js";

describe("Configuration", () => {
  it("should initialize configuration from an empty object", () => {
    const emptyConfig = {};
    const result = ensureConfiguration(emptyConfig);
    expect(result).toBeDefined();
    expect(typeof result).toBe("object");
  });
});
