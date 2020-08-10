from random import uniform, sample
from numpy import *
from copy import deepcopy


class TransE:
    def __init__(self, entityList, attrList, valueList, tripleList, EA_map, margin=1, learingRate=0.001, dim=200, L1=True):
        self.margin = margin
        self.learingRate = learingRate
        self.dim = dim  # 向量维度
        self.entityList = entityList  # 一开始，entityList是entity的list；初始化后，变为字典，key是entity，values是其向量（使用narray）。
        self.attrList = attrList  # 理由同上
        self.valueList = valueList
        self.tripleList = tripleList  # 理由同上
        self.EA_map = EA_map
        self.loss = 0
        self.L1 = L1

    def initialize(self):
        '''
        初始化向量
        '''
        entityVectorList = {}
        attrVectorList = {}
        valueVectorList = {}
        for entity in self.entityList:
            n = 0
            entityVector = []
            while n < self.dim:
                ram = init(self.dim)  # 初始化的范围
                entityVector.append(ram)
                n += 1
            entityVector = norm(entityVector)  # 归一化
            entityVectorList[entity] = entityVector
        print("entityVector初始化完成，数量是%d" % len(entityVectorList))
        for attr in self.attrList:
            n = 0
            attrVector = []
            while n < self.dim:
                ram = init(self.dim)  # 初始化的范围
                attrVector.append(ram)
                n += 1
            attrVector = norm(attrVector)  # 归一化
            attrVectorList[attr] = attrVector
        print("attrVectorList初始化完成，数量是%d" % len(attrVectorList))
        for value in self.valueList:
            n = 0
            valueVector = []
            while n < self.dim:
                ram = init(self.dim)  # 初始化的范围
                valueVector.append(ram)
                n += 1
            valueVector = norm(valueVector)  # 归一化
            valueVectorList[value] = valueVector
        print("valueVectorList初始化完成，数量是%d" % len(valueVectorList))
        for key in self.EA_map:
            entityVectorList[self.EA_map[key]] = entityVectorList[key]
        self.entityList = entityVectorList
        self.attrList = attrVectorList
        self.valueList = valueVectorList

    def transE(self, cI=20):
        print("训练开始")
        for cycleIndex in range(cI):
            Sbatch = self.getSample(550)
            Tbatch = []  # 元组对（原三元组，打碎的三元组）的列表 ：{((h,r,t),(h',r,t'))}
            for sbatch in Sbatch:
                tripletWithCorruptedTriplet = (sbatch, self.getCorruptedTriplet(sbatch))
                if(tripletWithCorruptedTriplet not in Tbatch):
                    Tbatch.append(tripletWithCorruptedTriplet)
            self.update(Tbatch)
            # if cycleIndex % 100 == 0:
            print("第%d次循环" % cycleIndex)
            print(self.loss)
            # self.writeRelationVector("c:\\relationVector.txt")
            # self.writeEntilyVector("c:\\entityVector.txt")
            self.loss = 0

    def getSample(self, size):
        return sample(self.tripleList, size)

    def getCorruptedTriplet(self, triplet):
        '''
        training triplets with either the head or tail replaced by a random entity (but not both at the same time)
        :param triplet:
        :return corruptedTriplet:
        '''
        i = uniform(-1.5, 1.5)
        if i < -0.5:  # 小于0，打坏三元组的第一项
            while True:
                entityTemp = sample(self.entityList.keys(), 1)[0]
                if entityTemp != triplet[0]:
                    break
            corruptedTriplet = (entityTemp, triplet[1], triplet[2])
        elif i < 0.5:  # 大于等于0，打坏三元组的第二项
            while True:
                valueTemp = sample(self.valueList.keys(), 1)[0]
                if valueTemp != triplet[1]:
                    break
            corruptedTriplet = (triplet[0], valueTemp, triplet[2])
        else:
            while True:
                attrTemp = sample(self.attrList.keys(), 1)[0]
                if attrTemp != triplet[2]:
                    break
            corruptedTriplet = (triplet[0], triplet[1], attrTemp)
        return corruptedTriplet

    def update(self, Tbatch):
        copyEntityList = deepcopy(self.entityList)
        copyAttrList = deepcopy(self.attrList)
        copyValueList = deepcopy(self.valueList)

        for tripletWithCorruptedTriplet in Tbatch:
            EntityVector = copyEntityList[tripletWithCorruptedTriplet[0][0]]  # tripletWithCorruptedTriplet是原三元组和打碎的三元组的元组tuple
            ValueVector = copyValueList[tripletWithCorruptedTriplet[0][1]]
            AttrVector = copyAttrList[tripletWithCorruptedTriplet[0][2]]
            EntityVectorWithCorruptedTriplet = copyEntityList[tripletWithCorruptedTriplet[1][0]]
            ValueVectorWithCorruptedTriplet = copyValueList[tripletWithCorruptedTriplet[1][1]]
            AttrVectorWithCorruptedTriplet = copyAttrList[tripletWithCorruptedTriplet[1][2]]

            # tripletWithCorruptedTriplet是原三元组和打碎的三元组的元组tuple
            EntityVectorBeforeBatch = self.entityList[tripletWithCorruptedTriplet[0][0]]
            ValueVectorBeforeBatch = self.valueList[tripletWithCorruptedTriplet[0][1]]
            AttrVectorBeforeBatch = self.attrList[tripletWithCorruptedTriplet[0][2]]
            EntityVectorWithCorruptedTripletBeforeBatch = self.entityList[tripletWithCorruptedTriplet[1][0]]
            ValueVectorWithCorruptedTripletBeforeBatch = self.valueList[tripletWithCorruptedTriplet[1][1]]
            AttrVectorWithCorruptedTripletBeforeBatch = self.attrList[tripletWithCorruptedTriplet[1][2]]

            if self.L1:
                distTriplet = distanceL1(EntityVectorBeforeBatch, ValueVectorBeforeBatch, AttrVectorBeforeBatch)
                distCorruptedTriplet = distanceL1(EntityVectorWithCorruptedTripletBeforeBatch,
                                                  ValueVectorWithCorruptedTripletBeforeBatch,  AttrVectorWithCorruptedTripletBeforeBatch)
            else:
                distTriplet = distanceL2(EntityVectorBeforeBatch, ValueVectorBeforeBatch, AttrVectorBeforeBatch)
                distCorruptedTriplet = distanceL2(EntityVectorWithCorruptedTripletBeforeBatch,
                                                  ValueVectorWithCorruptedTripletBeforeBatch,  AttrVectorWithCorruptedTripletBeforeBatch)
            eg = self.margin + distTriplet - distCorruptedTriplet
            if eg > 0:  # [function]+ 是一个取正值的函数
                self.loss += eg
                if self.L1:
                    tempPositive = 2 * self.learingRate * \
                        (ValueVectorBeforeBatch - EntityVectorBeforeBatch - AttrVectorBeforeBatch)
                    tempNegtative = 2 * self.learingRate * \
                        (ValueVectorWithCorruptedTripletBeforeBatch -
                         EntityVectorWithCorruptedTripletBeforeBatch - AttrVectorWithCorruptedTripletBeforeBatch)
                    tempPositiveL1 = []
                    tempNegtativeL1 = []
                    for i in range(self.dim):  # 不知道有没有pythonic的写法（比如列表推倒或者numpy的函数）？
                        if tempPositive[i] >= 0:
                            tempPositiveL1.append(1)
                        else:
                            tempPositiveL1.append(-1)
                        if tempNegtative[i] >= 0:
                            tempNegtativeL1.append(1)
                        else:
                            tempNegtativeL1.append(-1)
                    tempPositive = array(tempPositiveL1)
                    tempNegtative = array(tempNegtativeL1)

                else:
                    tempPositive = 2 * self.learingRate * \
                        (ValueVectorBeforeBatch - EntityVectorBeforeBatch - AttrVectorBeforeBatch)
                    tempNegtative = 2 * self.learingRate * \
                        (ValueVectorWithCorruptedTripletBeforeBatch -
                         EntityVectorWithCorruptedTripletBeforeBatch - AttrVectorWithCorruptedTripletBeforeBatch)

                EntityVector = EntityVector + tempPositive
                ValueVector = ValueVector - tempPositive
                AttrVector = AttrVector + tempPositive - tempNegtative
                EntityVectorWithCorruptedTriplet = EntityVectorWithCorruptedTriplet - tempNegtative
                ValueVectorWithCorruptedTriplet = ValueVectorWithCorruptedTriplet + tempNegtative
                AttrVectorWithCorruptedTriplet = AttrVectorWithCorruptedTriplet-tempNegtative+tempPositive

                # 只归一化这几个刚更新的向量，而不是按原论文那些一口气全更新了
                copyEntityList[tripletWithCorruptedTriplet[0][0]] = norm(EntityVector)
                copyValueList[tripletWithCorruptedTriplet[0][1]] = norm(ValueVector)
                copyAttrList[tripletWithCorruptedTriplet[0][2]] = norm(AttrVector)
                copyEntityList[tripletWithCorruptedTriplet[1][0]] = norm(EntityVectorWithCorruptedTriplet)
                copyValueList[tripletWithCorruptedTriplet[1][1]] = norm(ValueVectorWithCorruptedTriplet)
                copyAttrList[tripletWithCorruptedTriplet[1][2]] = norm(AttrVectorWithCorruptedTriplet)

        self.entityList = copyEntityList
        self.attrList = copyAttrList
        self.valueList = copyValueList

    def writeEntilyVector(self, dir):
        print("写入实体")
        entityVectorFile = open(dir, 'w')
        # entid=[]
        # for i in self.entityList.keys():
        #     entid.append(i)
        # print(entid)
        for i in range(len(self.entityList)):
            entityVectorFile.write(str(self.entityList[str(i)].tolist()))
            entityVectorFile.write("\n")
        entityVectorFile.close()

    # def writeRelationVector(self, dir):
    #     print("写入关系")
    #     relationVectorFile = open(dir, 'w')
    #     for relation in self.relationList.keys():
    #         relationVectorFile.write(relation + "\t")
    #         relationVectorFile.write(str(self.relationList[relation].tolist()))
    #         relationVectorFile.write("\n")
    #     relationVectorFile.close()


def init(dim):
    return uniform(-6/(dim**0.5), 6/(dim**0.5))


def distanceL1(h, t, r):
    s = h + r - t
    sum = fabs(s).sum()
    return sum


def distanceL2(h, t, r):
    s = h + r - t
    sum = (s*s).sum()
    return sum


def norm(list):
    '''
    归一化
    :param 向量
    :return: 向量的平方和的开方后的向量
    '''
    var = linalg.norm(list)
    i = 0
    while i < len(list):
        list[i] = list[i]/var
        i += 1
    return array(list)


def openDetailsAndId(dir, sp="\t"):
    idNum = 0
    list = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            DetailsAndId = line.strip().split(sp)
            list.append(DetailsAndId[0])
            idNum += 1
    return idNum, list


def openTrain(dir, sp="\t"):
    num = 0
    list = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            triple = line.strip().split(sp)
            if(len(triple) < 3):
                continue
            list.append(tuple(triple))
            num += 1
    return num, list


def read_EA_list(EAfile):
    seeds_map = {}
    with open(EAfile, 'r', encoding='utf-8') as r:
        lines = r.readlines()
    for line in lines:
        line = line.strip()
        e1, e2 = line.split()
        if e1 in seeds_map:
            print('error,', e1)
        else:
            seeds_map[e1] = e2
    return seeds_map


if __name__ == '__main__':
    entityIdNum, entityList = openDetailsAndId("data/ja_en/ent_ids_all")
    attrIdNum, attrList = openDetailsAndId("data/ja_en/att2id_all")
    valueIdNum, valueList = openDetailsAndId('data/ja_en/att_value2id_all')
    EA_map = read_EA_list('data/ja_en/ref_ent_ids')
    tripleNum, tripleList = openTrain("data/ja_en/att_triple_all")
    print("打开TransE")
    transE = TransE(entityList, attrList, valueList, tripleList, EA_map, margin=1, dim=200)
    print("TranE初始化")
    transE.initialize()
    transE.transE(6000)
    # transE.writeRelationVector("c:\\relationVector.txt")
    transE.writeEntilyVector("data/ja_en/ATentsembed.txt")
# ent_ids_all = ent_ids_1! + ent_ids_2!
# ref_ent_ids!
# open ent_ids_all  att2id_all att_value2id_all att_triple_all ref_ent_ids
#           !                                                     !
# save ATentsembed.txt
