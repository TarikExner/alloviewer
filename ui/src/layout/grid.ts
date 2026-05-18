export function plateGridTemplate(colCount: number, min = "1.8rem", max = "2.25rem") {
  return `auto repeat(${colCount}, minmax(${min}, ${max}))`;
}

