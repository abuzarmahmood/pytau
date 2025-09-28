#!/usr/bin/env python3
"""
Update README.md with latest test results table.
"""

import os
import re
from pathlib import Path

def update_readme_with_tests():
    """Update README.md with test results table."""
    readme_path = Path('README.md')
    
    if not readme_path.exists():
        print("README.md not found!")
        return False
    
    # Read the test results table
    table_path = Path('test_results_table.md')
    if not table_path.exists():
        print("test_results_table.md not found!")
        return False
    
    with open(table_path, 'r') as f:
        test_table = f.read().strip()
    
    # Read current README
    with open(readme_path, 'r') as f:
        readme_content = f.read()
    
    # Define the test results section
    test_section = f"""## 🧪 Test Results

{test_table}

*Last updated: {os.environ.get('GITHUB_RUN_ID', 'manually')}*

"""
    
    # Look for existing test results section
    test_section_pattern = r'## 🧪 Test Results.*?(?=\n## |\n# |\Z)'
    
    if re.search(test_section_pattern, readme_content, re.DOTALL):
        # Replace existing section
        new_content = re.sub(test_section_pattern, test_section.rstrip(), readme_content, flags=re.DOTALL)
    else:
        # Find a good place to insert the test results section
        # Look for the Quick Start section and insert after it
        quick_start_pattern = r'(## ⚡ Quick Start.*?)(\n## )'
        match = re.search(quick_start_pattern, readme_content, re.DOTALL)
        
        if match:
            # Insert after Quick Start section
            new_content = readme_content[:match.end(1)] + '\n\n' + test_section + match.group(2) + readme_content[match.end(2)+1:]
        else:
            # If no Quick Start found, insert after the badges section
            badges_pattern = r'(.*?badge\.svg\)\]\(.*?\)\n)'
            match = re.search(badges_pattern, readme_content, re.DOTALL)
            
            if match:
                new_content = readme_content[:match.end(1)] + '\n' + test_section + readme_content[match.end(1):]
            else:
                # Fallback: insert after the first heading
                lines = readme_content.split('\n')
                insert_index = 1
                for i, line in enumerate(lines):
                    if line.startswith('## '):
                        insert_index = i
                        break
                
                lines.insert(insert_index, test_section.rstrip())
                new_content = '\n'.join(lines)
    
    # Write updated README
    with open(readme_path, 'w') as f:
        f.write(new_content)
    
    print("README.md updated with test results!")
    return True

def main():
    """Main function."""
    if update_readme_with_tests():
        print("Successfully updated README with test results")
    else:
        print("Failed to update README")
        exit(1)

if __name__ == "__main__":
    main()