# Simple script that demonstrates myCBR's external interface

import sys

def sim2(query, case):
    
    # special sim: returns 1.0 if the number of 'a' is the same in query and in
    #              case or if one is undefined; 0.0 otherwise
  
    if query == "_undefined_" or case == "_undefined_": return 1.0
    if len(query) > 0 and query[0] == '"':
        q = eval(query)                                # parse the string
    else:
        q = query
    if len(case) > 0 and case[0] == '"':
        c = eval(case)                                 # parse the string
    else:
        c = case
    if q.count('e') == c.count('e'):                   # if the number of 'e's is the same
        return 1.0
    else:
        return 0.0

def main():
    print str(sim2(sys.argv[1], sys.argv[2]))

main()
