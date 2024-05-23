import gosm.mesh as mesh

m = mesh.CubeMesh([0,0],2000,100)
print(m)

m2= m.feasible_mesh(80,80)
m3 = mesh.Mesh([cell for cell in m.cells
                         if sum(cell.lower_left) <= 80])

print(m2)
print(m3)
print('m2',len(m2.cells), 'm3',len(m3.cells))